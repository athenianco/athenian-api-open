# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -mavx2
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

from posix.dlfcn cimport RTLD_LAZY, dlclose, dlopen, dlsym

cimport cython
from cpython cimport Py_INCREF, PyObject
from cython.operator cimport dereference, postincrement
from libc.stdint cimport int8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport memcpy, memset, strncmp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search
from libcpp.utility cimport pair
from numpy cimport (
    NPY_ARRAY_C_CONTIGUOUS,
    NPY_UINT,
    PyArray_BYTES,
    PyArray_CheckExact,
    PyArray_DescrFromType,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    dtype as np_dtype,
    import_array,
    ndarray,
    npy_intp,
)

from athenian.api.native.cpython cimport (
    Py_None,
    PyBytes_FromStringAndSize,
    PyList_GET_ITEM,
    PyList_New,
    PyList_SET_ITEM,
    PyLong_AsLong,
    PyLong_FromLong,
    PyTuple_GET_ITEM,
    PyUnicode_DATA,
    PyUnicode_FromStringAndSize,
    PyUnicode_GET_LENGTH,
    PyUnicode_New,
)
from athenian.api.native.mi_heap_destroy_stl_allocator cimport (
    mi_heap_allocator_from_capsule,
    mi_heap_destroy_stl_allocator,
    mi_unordered_map,
    mi_unordered_set,
    mi_vector,
)
from athenian.api.native.numpy cimport (
    PyArray_Descr,
    PyArray_DESCR,
    PyArray_DescrNew,
    PyArray_NewFromDescr,
    PyArray_Type,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.string_view cimport string_view

from typing import Any, Sequence

import asyncpg
import numpy as np


cdef extern from "../../../asyncpg_recordobj.h" nogil:
    PyObject *ApgRecord_GET_ITEM(PyObject *, int)
    void ApgRecord_SET_ITEM(object, int, object)
    PyObject *ApgRecord_GET_DESC(PyObject *)


cdef extern from "dag_accelerated.h" nogil:
    size_t sorted_set_difference_avx2(
        const uint32_t *set1,
        const size_t length1,
        const uint32_t *set2,
        const size_t length2,
        uint32_t *out,
    )


# ApgRecord_New is not exported from the Python interface of asyncpg
ctypedef object (*_ApgRecord_New)(type, PyObject *, Py_ssize_t)
cdef _ApgRecord_New ApgRecord_New
cdef void *_self = dlopen(NULL, RTLD_LAZY)
ApgRecord_New = <_ApgRecord_New>dlsym(_self, "ApgRecord_New")
dlclose(_self)
import_array()


def searchsorted_inrange(ndarray a, v: Any, side="left", sorter=None):
    r = np.searchsorted(a, np.atleast_1d(v), side=side, sorter=sorter)
    r[r == len(a)] = 0  # whatever index is fine
    return r


def extract_subdag(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray heads,
    *,
    return_indexes=False,
    alloc_capsule=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(vertexes) == len(hashes) + 1
    assert heads.dtype.char == "S"
    if len(hashes) == 0:
        if return_indexes:
            hashes = np.arange(len(hashes), dtype=np.uint32)
        return hashes, vertexes, edges
    if len(heads):
        heads = np.sort(heads)
        existing_heads = searchsorted_inrange(hashes, heads)
        existing_heads = existing_heads[hashes[existing_heads] == heads].astype(np.uint32)
    else:
        existing_heads = np.array([], dtype=np.uint32)
    left_vertexes_map = np.zeros_like(vertexes)
    left_vertexes = np.zeros_like(vertexes)
    left_edges = np.zeros_like(edges)
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        long left_count
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] existing_heads_view = existing_heads
        uint32_t[:] left_vertexes_map_view = left_vertexes_map
        uint32_t[:] left_vertexes_view = left_vertexes
        uint32_t[:] left_edges_view = left_edges
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        left_count = _extract_subdag(
            vertexes_view,
            edges_view,
            existing_heads_view,
            False,
            left_vertexes_map_view,
            left_vertexes_view,
            left_edges_view,
            dereference(alloc),
        )
    if return_indexes:
        left_hashes = left_vertexes_map[:left_count]
    else:
        left_hashes = hashes[left_vertexes_map[:left_count]]
    left_vertexes = left_vertexes[:left_count + 1]
    left_edges = left_edges[:left_vertexes[left_count]]
    return left_hashes, left_vertexes, left_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint32_t _extract_subdag(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] heads,
    bool only_map,
    uint32_t[:] left_vertexes_map,
    uint32_t[:] left_vertexes,
    uint32_t[:] left_edges,
    mi_heap_destroy_stl_allocator[char] alloc,
) nogil:
    cdef:
        optional[mi_vector[uint32_t]] boilerplate
        uint32_t i, j, head, peek, edge
        const uint32_t *raw_vertexes = &vertexes[0]
        const uint32_t *raw_edges = &edges[0]
        uint32_t *raw_left_vertexes_map = &left_vertexes_map[0]
    boilerplate.emplace(alloc)
    dereference(boilerplate).reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(heads)):
        head = heads[i]
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if raw_left_vertexes_map[peek]:
                continue
            raw_left_vertexes_map[peek] = 1
            for j in range(raw_vertexes[peek], raw_vertexes[peek + 1]):
                edge = raw_edges[j]
                if not raw_left_vertexes_map[edge]:
                    dereference(boilerplate).push_back(edge)
    if only_map:
        return 0
    # compress the vertex index mapping
    cdef uint32_t left_count = 0, edge_index, v
    for i in range(len(left_vertexes_map)):
        if raw_left_vertexes_map[i]:
            raw_left_vertexes_map[i] = left_count + 1  # disambiguate 0, become 1-based indexed
            left_count += 1
    # len(left_vertexes) == 0 means we don't care about the extracted edges
    if len(left_vertexes) > 0:
        # rebuild the edges
        edge_index = 0
        for i in range(len(vertexes) - 1):
            v = raw_left_vertexes_map[i]
            if v:
                v -= 1  # become 0-based indexed again
                left_vertexes[v] = edge_index
                for j in range(raw_vertexes[i], raw_vertexes[i + 1]):
                    edge = raw_edges[j]
                    left_edges[edge_index] = raw_left_vertexes_map[edge] - 1
                    edge_index += 1
        left_vertexes[left_count] = edge_index
    # invert the vertex index mapping
    left_count = 0
    for i in range(len(left_vertexes_map)):
        if raw_left_vertexes_map[i]:
            raw_left_vertexes_map[left_count] = i
            left_count += 1
    return left_count


def sorted_set_difference(
    ndarray arr1 not None,
    ndarray arr2 not None,
) -> ndarray:
    assert PyArray_NDIM(arr1) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr1)
    assert PyArray_DESCR(<PyObject *> arr1).kind == b"u"
    assert PyArray_NDIM(arr2) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr2)
    assert PyArray_DESCR(<PyObject *> arr2).kind == b"u"

    cdef:
        uint32_t *arr1_data = <uint32_t *> PyArray_BYTES(arr1)
        uint32_t *arr2_data = <uint32_t *> PyArray_BYTES(arr2)
        npy_intp len1 = PyArray_DIM(arr1, 0)
        npy_intp len2 = PyArray_DIM(arr2, 0)
        ndarray output
        np_dtype u32dtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_UINT))

    output = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> u32dtype,
        1,
        &len1 if len1 > len2 else &len2,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(u32dtype)
    output.shape[0] = sorted_set_difference_avx2(
        arr1_data,
        len1,
        arr2_data,
        len2,
        <uint32_t *> PyArray_BYTES(output),
    )
    return output


cdef struct Edge:
    uint32_t vertex
    uint32_t position


def join_dags(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    list new_edges,
    alloc_capsule=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cdef:
        Py_ssize_t size
        long i, hpos, parent_index
        const char *parent_oid
        const char *child_oid
        PyObject *record
        PyObject *obj
        char *new_hashes_data
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_map[string_view, int]] new_hashes_map, hashes_map
        mi_unordered_map[string_view, int].iterator it
        mi_vector[Edge] *found_edges
        Edge edge
        bool exists
    size = len(new_edges)
    if size == 0:
        return hashes, vertexes, edges
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    new_hashes_map.emplace(dereference(alloc))
    hashes_map.emplace(dereference(alloc))
    new_hashes_arr = np.empty(size * 2, dtype="S40")
    new_hashes_data = PyArray_BYTES(new_hashes_arr)
    hpos = 0
    if isinstance(new_edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                parent_oid = <const char*>PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                memcpy(new_hashes_data + hpos, parent_oid, 40)
                hpos += 40
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    memcpy(new_hashes_data + hpos, child_oid, 40)
                    hpos += 40
    else:
        assert isinstance(new_edges[0], tuple)
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                memcpy(new_hashes_data + hpos, parent_oid, 40)
                hpos += 40
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    memcpy(new_hashes_data + hpos, child_oid, 40)
                    hpos += 40
    new_hashes_arr = new_hashes_arr[:hpos // 40]
    if len(hashes) > 0:
        new_hashes = np.unique(np.concatenate([new_hashes_arr, hashes]))
        found_matches = np.searchsorted(hashes, new_hashes)
        found_matches_in_range = found_matches.copy()
        found_matches_in_range[found_matches == len(hashes)] = 0
        distinct_mask = hashes[found_matches_in_range] != new_hashes
        found_matches = found_matches[distinct_mask]
        new_hashes = new_hashes[distinct_mask]
        result_hashes = np.insert(hashes, found_matches, new_hashes)
    else:
        new_hashes = np.unique(new_hashes_arr)
        found_matches = np.array([], dtype=int)
        result_hashes = new_hashes

    size = len(new_hashes)
    new_hashes_data = PyArray_BYTES(new_hashes)
    with nogil:
        for i in range(size):
            dereference(new_hashes_map)[string_view(new_hashes_data + i * 40, 40)] = i
    if len(hashes) > 0:
        size = len(result_hashes)
        new_hashes_data = PyArray_BYTES(result_hashes)
        with nogil:
            for i in range(size):
                dereference(hashes_map)[string_view(new_hashes_data + i * 40, 40)] = i
    else:
        hashes_map = new_hashes_map

    cdef:
        optional[mi_vector[mi_vector[Edge]]] new_edges_lists
        long new_edges_counter = 0
    new_edges_lists.emplace(dereference(alloc))
    dereference(new_edges_lists).reserve(dereference(new_hashes_map).size())
    for _ in range(dereference(new_hashes_map).size()):
        dereference(new_edges_lists).emplace_back(dereference(alloc))
    size = len(new_edges)
    if isinstance(new_edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if not strncmp(child_oid, "0" * 40, 40):
                    # initial commit
                    continue
                parent_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                it = dereference(new_hashes_map).find(string_view(parent_oid, 40))
                if it != dereference(new_hashes_map).end():
                    parent_index = PyLong_AsLong(ApgRecord_GET_ITEM(record, 2))
                    found_edges = &dereference(new_edges_lists)[dereference(it).second]
                    exists = False
                    for j in range(<int>found_edges.size()):
                        if <int>dereference(found_edges)[j].position == parent_index:
                            exists = True
                            break
                    if not exists:
                        # https://github.com/cython/cython/issues/1642
                        edge.vertex = dereference(hashes_map)[string_view(child_oid, 40)]
                        edge.position = parent_index
                        found_edges.push_back(edge)
                        new_edges_counter += 1
    else:
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if not strncmp(child_oid, "0" * 40, 40):
                    # initial commit
                    continue
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                it = dereference(new_hashes_map).find(string_view(parent_oid, 40))
                if it != dereference(new_hashes_map).end():
                    parent_index = PyLong_AsLong(PyTuple_GET_ITEM(record, 2))
                    found_edges = &dereference(new_edges_lists)[dereference(it).second]
                    exists = False
                    for j in range(<int>found_edges.size()):
                        if <int>dereference(found_edges)[j].position == parent_index:
                            exists = True
                            break
                    if not exists:
                        # https://github.com/cython/cython/issues/1642
                        edge.vertex = dereference(hashes_map)[string_view(child_oid, 40)]
                        edge.position = parent_index
                        found_edges.push_back(edge)
                        new_edges_counter += 1

    old_vertex_map = np.zeros(len(hashes), dtype=np.uint32)
    result_vertexes = np.zeros(len(result_hashes) + 1, dtype=np.uint32)
    result_edges = np.zeros(len(edges) + new_edges_counter, dtype=np.uint32)
    if len(hashes) > 0:
        found_matches += np.arange(len(found_matches))
    cdef:
        const int64_t[:] found_matches_view = found_matches
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        uint32_t[:] old_vertex_map_view = old_vertex_map
        uint32_t[:] result_vertexes_view = result_vertexes
        uint32_t[:] result_edges_view = result_edges
    with nogil:
        _recalculate_vertices_and_edges(
            found_matches_view, vertexes_view, edges_view, &dereference(new_edges_lists),
            old_vertex_map_view, result_vertexes_view, result_edges_view)
    return result_hashes, result_vertexes, result_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _recalculate_vertices_and_edges(
    const int64_t[:] found_matches,
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const mi_vector[mi_vector[Edge]] *new_edges_lists,
    uint32_t[:] old_vertex_map,
    uint32_t[:] result_vertexes,
    uint32_t[:] result_edges,
) nogil:
    cdef:
        uint32_t j, left, offset = 0, pos = 0, list_size
        uint32_t old_edge_i = 0, new_edge_i = 0, size = len(result_vertexes) - 1, i
        const mi_vector[Edge] *new_edge_list
        bint has_old = len(vertexes) > 1
        Edge edge
    if has_old:
        # populate old_vertex_map
        for i in range(size):
            if offset >= len(found_matches) or i < found_matches[offset]:
                old_vertex_map[i - offset] = i
            else:
                offset += 1
    # write the edges
    for i in range(size):
        result_vertexes[i] = pos
        if (new_edge_i >= len(found_matches) or i < found_matches[new_edge_i]) and has_old:
            # write old edge
            left = vertexes[old_edge_i]
            offset = vertexes[old_edge_i + 1] - left
            for j in range(offset):
                result_edges[pos + j] = old_vertex_map[edges[left + j]]
            pos += offset
            old_edge_i += 1
        else:
            new_edge_list = &dereference(new_edges_lists)[new_edge_i]
            list_size = new_edge_list.size()
            for j in range(list_size):
                edge = dereference(new_edge_list)[j]
                result_edges[pos + edge.position] = edge.vertex
            pos += list_size
            new_edge_i += 1
    result_vertexes[size] = pos


@cython.boundscheck(False)
def append_missing_heads(
    list edges,
    ndarray hashes,
    alloc_capsule=None,
) -> None:
    cdef:
        optional[mi_heap_destroy_stl_allocator[string_view]] alloc
        optional[mi_unordered_set[string_view]] hashes_set
        mi_unordered_set[string_view].const_iterator it
        Py_ssize_t size
        long i
        PyObject *record
        PyObject *desc = Py_None
        object new_record, elem
        const char *hashes_data
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    size = len(hashes)
    hashes_data = PyArray_BYTES(hashes)
    with nogil:
        hashes_set.emplace(dereference(alloc))
        for i in range(size):
            dereference(hashes_set).emplace(hashes_data + 40 * i, 40)
    size = len(edges)
    if size > 0:
        if isinstance(edges[0], asyncpg.Record):
            with nogil:
                for i in range(size):
                    record = PyList_GET_ITEM(<PyObject *>edges, i)
                    dereference(hashes_set).erase(string_view(
                        <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0)), 40))
                    dereference(hashes_set).erase(string_view(
                        <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1)), 40))
                desc = ApgRecord_GET_DESC(PyList_GET_ITEM(<PyObject *>edges, 0))
        else:
            assert isinstance(edges[0], tuple)
            with nogil:
                for i in range(size):
                    record = PyList_GET_ITEM(<PyObject *>edges, i)
                    dereference(hashes_set).erase(string_view(
                        <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0)), 40))
                    dereference(hashes_set).erase(string_view(
                        <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1)), 40))
    it = dereference(hashes_set).const_begin()
    while it != dereference(hashes_set).const_end():
        if desc != Py_None:
            new_record = ApgRecord_New(asyncpg.Record, desc, 3)
            elem = PyUnicode_FromStringAndSize(dereference(it).data(), 40)
            Py_INCREF(elem)
            ApgRecord_SET_ITEM(new_record, 0, elem)
            elem = PyUnicode_FromStringAndSize("0" * 40, 40)
            Py_INCREF(elem)
            ApgRecord_SET_ITEM(new_record, 1, elem)
            Py_INCREF(0)  # interned
            ApgRecord_SET_ITEM(new_record, 2, 0)
        else:
            new_record = (PyUnicode_FromStringAndSize(dereference(it).data(), 40), "0" * 40, 0)
        edges.append(new_record)
        postincrement(it)


ctypedef pair[int, const char *] RawEdge


@cython.boundscheck(False)
def verify_edges_integrity(list edges, ignored=None, alloc_capsule=None) -> tuple[list[int], list[int], ndarray]:
    cdef:
        Py_ssize_t size = len(edges)
        Py_ssize_t children_size
        const char *oid
        long indexes_sum, parent_index, i, j, is_asyncpg, ignored_size = 0
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_set[string_view]] ignored_set
        const char *ignored_data = NULL
        optional[mi_unordered_map[string_view, mi_vector[RawEdge]]] children_indexes
        mi_unordered_map[string_view, mi_vector[RawEdge]].iterator it
        PyObject *record
        PyObject *obj
        mi_vector[RawEdge] *children_range
        optional[mi_vector[char]] bads
        optional[mi_unordered_map[string_view, int]] edge_parent_map
        mi_unordered_map[string_view, int].iterator edge_parent_map_it, edge_parent_map_end
        optional[mi_unordered_map[string_view, mi_vector[int]]] reversed_edges
        mi_unordered_map[string_view, mi_vector[int]].iterator parent_it
        optional[mi_vector[const char *]] edge_parents
        optional[mi_vector[int]] boilerplate
        string_view tmpstr
        int *edges_data
        size_t edge_i
        bool exists
        list tainted_indexes
        list bad_seeds
        char bad_kind
        ndarray tainted_hashes
        char *tainted_hashes_data

    if size == 0:
        return [], [], np.array([], dtype="S40")
    is_asyncpg = isinstance(edges[0], asyncpg.Record)
    if not is_asyncpg:
        assert isinstance(edges[0], tuple)
    if ignored is not None:
        assert PyArray_CheckExact(ignored)
        assert PyArray_NDIM(ignored) == 1
        assert PyArray_IS_C_CONTIGUOUS(ignored)
        assert ignored.dtype == "S40"
        ignored_data = PyArray_BYTES(ignored)
        ignored_size = PyArray_DIM(ignored, 0)

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()

    with nogil:
        ignored_set.emplace(dereference(alloc))
        bads.emplace(size, dereference(alloc))
        edge_parents.emplace(size, dereference(alloc))
        children_indexes.emplace(dereference(alloc))
        edge_parent_map.emplace(dereference(alloc))
        reversed_edges.emplace(dereference(alloc))

        if <PyObject *>ignored != Py_None:
            for i in range(ignored_size):
                dereference(ignored_set).emplace(ignored_data + i * 40, 40)

        if is_asyncpg:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                obj = ApgRecord_GET_ITEM(record, 0)
                if obj == Py_None:
                    dereference(bads)[i] = 1
                    continue
                if PyUnicode_GET_LENGTH(obj) != 40:
                    dereference(bads)[i] = 1
                    continue
                oid = <const char *> PyUnicode_DATA(obj)
                tmpstr = string_view(oid, 40)
                dereference(edge_parent_map)[tmpstr] = i
                dereference(edge_parents)[i] = oid
                parent_index = PyLong_AsLong(ApgRecord_GET_ITEM(record, 2))
                children_range = &dereference(dereference(children_indexes).try_emplace(tmpstr, dereference(alloc)).first).second

                obj = ApgRecord_GET_ITEM(record, 1)
                if obj == Py_None:
                    dereference(bads)[i] = 1
                    continue
                if PyUnicode_GET_LENGTH(obj) != 40:
                    dereference(bads)[i] = 1
                    continue
                oid = <const char *> PyUnicode_DATA(obj)

                exists = False
                for j in range(<int>children_range.size()):
                    if dereference(children_range)[j].first == parent_index:
                        exists = True
                        if strncmp(dereference(children_range)[j].second, oid, 40):
                            dereference(bads)[i] = 1
                        break
                dereference(dereference(reversed_edges).try_emplace(string_view(oid, 40), dereference(alloc)).first).second.push_back(i)
                if not exists:
                    children_range.push_back(RawEdge(parent_index, oid))
        else:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                obj = PyTuple_GET_ITEM(record, 0)
                if obj == Py_None:
                    dereference(bads)[i] = 1
                    continue
                if PyUnicode_GET_LENGTH(obj) != 40:
                    dereference(bads)[i] = 1
                    continue
                oid = <const char *> PyUnicode_DATA(obj)
                tmpstr = string_view(oid, 40)
                dereference(edge_parent_map)[tmpstr] = i
                dereference(edge_parents)[i] = oid
                parent_index = PyLong_AsLong(PyTuple_GET_ITEM(record, 2))
                children_range = &dereference(dereference(children_indexes).try_emplace(tmpstr, dereference(alloc)).first).second

                obj = PyTuple_GET_ITEM(record, 1)
                if obj == Py_None:
                    dereference(bads)[i] = 1
                    continue
                if PyUnicode_GET_LENGTH(obj) != 40:
                    dereference(bads)[i] = 1
                    continue
                oid = <const char *> PyUnicode_DATA(obj)

                exists = False
                for j in range(<int>children_range.size()):
                    if dereference(children_range)[j].first == parent_index:
                        exists = True
                        if strncmp(dereference(children_range)[j].second, oid, 40):
                            dereference(bads)[i] = 1
                        break
                dereference(dereference(reversed_edges).try_emplace(string_view(oid, 40), dereference(alloc)).first).second.push_back(i)
                if not exists:
                    children_range.push_back(RawEdge(parent_index, oid))

        it = dereference(children_indexes).begin()
        while it != dereference(children_indexes).end():
            children_range = &dereference(it).second
            children_size = children_range.size()
            indexes_sum = 0
            for i in range(children_size):
                indexes_sum += dereference(children_range)[i].first
            indexes_sum -= ((children_size - 1) * children_size) // 2
            if indexes_sum != 0:
                dereference(bads)[dereference(edge_parent_map)[dereference(it).first]] = 1
            postincrement(it)

        # propagate up
        dereference(edge_parent_map).clear()  # we will store tainted hashes here
        boilerplate.emplace(dereference(alloc))
        for i in range(size):
            if dereference(bads)[i]:
                dereference(boilerplate).push_back(i)
                while not dereference(boilerplate).empty():
                    parent_index = dereference(boilerplate).back()
                    dereference(boilerplate).pop_back()
                    oid = dereference(edge_parents)[parent_index]
                    if oid == NULL:
                        continue
                    tmpstr = string_view(oid, 40)
                    if dereference(ignored_set).count(tmpstr) > 0:
                        continue
                    dereference(edge_parent_map)[tmpstr] = True
                    parent_it = dereference(reversed_edges).find(tmpstr)
                    if parent_it != dereference(reversed_edges).end():
                        edges_data = dereference(parent_it).second.data()
                        for edge_i in range(dereference(parent_it).second.size()):
                            j = edges_data[edge_i]
                            if not dereference(bads)[j]:
                                dereference(boilerplate).push_back(j)
                                dereference(bads)[j] = 2

    tainted_indexes = []
    bad_seeds = []
    edge_parent_map_end = dereference(edge_parent_map).end()
    for i in range(size):
        oid = dereference(edge_parents)[i]
        if oid != NULL:
            tmpstr = string_view(oid, 40)
        bad_kind = dereference(bads)[i]
        if (
            bad_kind
            or
            oid != NULL and dereference(edge_parent_map).find(tmpstr) != edge_parent_map_end
        ):
            if bad_kind < 2 and (oid == NULL or dereference(ignored_set).count(string_view(oid, 40)) == 0):
                bad_seeds.append(i)
            # we should always remove the bad seeds to not break downstream
            tainted_indexes.append(i)

    tainted_hashes = np.empty(dereference(edge_parent_map).size(), dtype="S40")
    tainted_hashes_data = PyArray_BYTES(tainted_hashes)
    edge_parent_map_it = dereference(edge_parent_map).begin()
    i = 0
    while edge_parent_map_it != edge_parent_map_end:
        memcpy(tainted_hashes_data + i, dereference(edge_parent_map_it).first.data(), 40)
        i += 40
        postincrement(edge_parent_map_it)

    return tainted_indexes, bad_seeds, tainted_hashes


ctypedef char sha_t[40]


cdef inline bool _compare_shas(const char *first, const char *second) nogil:
    return strncmp(first, second, 40) < 0


@cython.boundscheck(False)
def find_orphans(
    list edges,
    ndarray attach_to,
    alloc_capsule=None,
) -> dict[str, list[int]]:
    cdef:
        Py_ssize_t size = len(edges), i
        const char *child_oid
        const char *parent_oid
        bint is_asyncpg
        size_t j
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_set[string_view]] parents
        optional[mi_unordered_map[string_view, mi_vector[RawEdge]]] reversed_edges
        mi_unordered_map[string_view, mi_vector[RawEdge]].iterator reversed_parents
        sha_t *attach_data
        Py_ssize_t attach_length = len(attach_to)
        PyObject *record
        optional[mi_vector[RawEdge]] leaves
        optional[mi_vector[RawEdge]] boilerplate
        RawEdge edge, root_edge
        RawEdge *leaves_data
        RawEdge *reversed_data
        optional[mi_vector[mi_unordered_set[int]]] rejected

    if size == 0:
        return {}

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    parents.emplace(dereference(alloc))
    reversed_edges.emplace(dereference(alloc))
    leaves.emplace(dereference(alloc))
    boilerplate.emplace(dereference(alloc))
    rejected.emplace(dereference(alloc))
    assert attach_to.dtype == "S40"
    attach_data = <sha_t *>PyArray_BYTES(attach_to)
    is_asyncpg = isinstance(edges[0], asyncpg.Record)
    with nogil:
        if is_asyncpg:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                parent_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if strncmp(child_oid, b"0" * 40, 40):
                    dereference(parents).emplace(parent_oid, 40)
                    dereference(leaves).emplace_back(i, child_oid)
                    dereference(dereference(reversed_edges).try_emplace(
                        string_view(child_oid, 40), dereference(alloc),
                    ).first).second.emplace_back(i, parent_oid)
                else:
                    dereference(leaves).emplace_back(i, parent_oid)
        else:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *> edges, i)
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if strncmp(child_oid, b"0" * 40, 40):
                    dereference(parents).emplace(parent_oid, 40)
                    dereference(leaves).emplace_back(i, child_oid)
                    dereference(dereference(reversed_edges).try_emplace(
                        string_view(child_oid, 40), dereference(alloc),
                    ).first).second.emplace_back(i, parent_oid)
                else:
                    dereference(leaves).emplace_back(i, parent_oid)

        # propagate orphaned leaves up, recording the parents in `rejected`
        leaves_data = dereference(leaves).data()
        size = 0
        for i in range(<Py_ssize_t> dereference(leaves).size()):
            if dereference(parents).find(string_view(leaves_data[i].second, 40)) == dereference(parents).end():
                if not binary_search(
                    attach_data,
                    attach_data + attach_length,
                    dereference(<sha_t *>leaves_data[i].second),
                    _compare_shas,
                ):
                    root_edge = leaves_data[i]
                    # we must set index to -1 to be not exist the cycle after the first iteration
                    dereference(boilerplate).emplace_back(-1, root_edge.second)
                    leaves_data[size] = root_edge
                    size += 1
                    dereference(rejected).emplace_back(dereference(alloc))
                    while not dereference(boilerplate).empty():
                        edge = dereference(boilerplate).back()
                        dereference(boilerplate).pop_back()
                        if edge.first >= 0 and not dereference(rejected).back().emplace(edge.first).second:
                            continue
                        reversed_parents = dereference(reversed_edges).find(string_view(edge.second, 40))
                        if reversed_parents != dereference(reversed_edges).end():
                            reversed_data = dereference(reversed_parents).second.data()
                            for j in range(dereference(reversed_parents).second.size()):
                                dereference(boilerplate).emplace_back(reversed_data[j])
                    # in case hash -> 0000000000000000000000000000000000000000
                    dereference(rejected).back().emplace(root_edge.first)

    result = {
        PyUnicode_FromStringAndSize(leaves_data[i].second, 40):
            _unordered_set_to_list(dereference(rejected)[i])
        for i in range(size)
    }
    return result


cdef inline list _unordered_set_to_list(mi_unordered_set[int] &items):
    cdef:
        PyObject *result = PyList_New(items.size())
        Py_ssize_t i = 0
        mi_unordered_set[int].iterator it = items.begin()
    while it != items.end():
        PyList_SET_ITEM(result, i, PyLong_FromLong(dereference(it)))
        i += 1
        postincrement(it)
    return <list> result


def mark_dag_access(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray heads,
    heads_order_is_significant: bool,
    alloc_capsule=None,
) -> np.ndarray:
    """
    Find the earliest parent from `heads` for each commit in `hashes`.

    If `heads_order_is_significant`, the `heads` must be sorted by commit timestamp in descending \
    order. Thus `heads[0]` should be the latest commit.

    If not `heads_order_is_significant`, we sort `heads` topologically, but the earlier commits \
    have the priority over the later commits, if they are the same.

    :return: Indexes in `heads`, *not vertexes*. `len(heads)` if there is no parent.
    """
    if len(hashes) == 0:
        return np.array([], dtype=np.int64)
    size = len(heads)
    access = np.full(len(vertexes), size, np.int32)
    if size == 0:
        return access[:-1]
    assert heads.dtype.char == "S"
    # we cannot sort heads because the order is important - we return the original indexes
    existing_heads = searchsorted_inrange(hashes, heads)
    matched = hashes[existing_heads] == heads
    head_vertexes = np.full(size + 1, len(vertexes), np.uint32)
    head_vertexes[:-1][matched] = existing_heads[matched]
    heads = head_vertexes
    del head_vertexes
    if not matched.any():
        return access[:-1]
    order = np.full(size, size, np.int32)
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        bool heads_order_is_significant_native = heads_order_is_significant
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_without_tail = heads[:-1]
        const uint32_t[:] heads_view = heads
        int32_t[:] order_view = order
        int32_t[:] access_view = access
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        _toposort(
            vertexes_view,
            edges_view,
            heads_without_tail,
            heads_order_is_significant_native,
            order_view,
            dereference(alloc),
        )
        _mark_dag_access(
            vertexes_view,
            edges_view,
            heads_view,
            order_view,
            access_view,
            dereference(alloc),
        )
    return access[:-1]  # len(vertexes) = len(hashes) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _toposort(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] heads,
    bool heads_order_is_significant,
    int32_t[:] order,
    mi_heap_destroy_stl_allocator[char] &alloc,
) nogil:
    """Topological sort of `heads`. The order is reversed!"""
    cdef:
        optional[mi_vector[uint32_t]] boilerplate
        optional[mi_vector[int32_t]] visited_alloc
        int32_t *visited
        uint32_t j, head, peek, edge, missing = len(vertexes)
        int64_t i, order_pos = 0
        int32_t status, size = len(heads), vv = size + 1

    boilerplate.emplace(alloc)
    dereference(boilerplate).reserve(max(1, len(edges) - len(vertexes) + 1))
    visited_alloc.emplace(len(vertexes), alloc)
    visited = dereference(visited_alloc).data()
    for i in range(len(heads)):
        head = heads[i]
        if head == missing:
            continue
        visited[head] = i - size  # fused head marks in `visited`  array
    # heads, unvisited -> -len(heads), ..., -2, -1
    # normal vertexes, unvisited -> 0
    # heads, visited -> 1, 2, ..., len(heads)
    # normal vertexes, visited -> len(heads) + 1
    for i in range(len(heads) - 1, -1, -1):  # reverse order is release-friendly
        # we start from the earliest head and end at the latest
        head = heads[i]
        if head == missing:
            continue
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            status = visited[peek]
            if status > 0:
                dereference(boilerplate).pop_back()
                if status < vv:
                    status -= 1  # index of the head
                    if status >= i or not heads_order_is_significant:
                        # status >= i means it comes after => appeared earlier
                        # we must ignore future releases standing in front
                        order[order_pos] = status
                        order_pos += 1
                    visited[peek] = vv
                continue
            visited[peek] += vv
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if visited[edge] <= 0:
                    dereference(boilerplate).push_back(edge)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _mark_dag_access(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] heads,
    const int32_t[:] order,
    int32_t[:] access,
    mi_heap_destroy_stl_allocator[char] &alloc,
) nogil:
    cdef:
        optional[mi_vector[uint32_t]] boilerplate
        uint32_t j, head, peek, edge, missing = len(vertexes)
        int64_t i, original_index
        int32_t size = len(order)
    boilerplate.emplace(alloc)
    dereference(boilerplate).reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(size):
        head = heads[order[i]]
        if head == missing:
            continue
        original_index = order[i]
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if access[peek] < size:
                continue
            access[peek] = original_index
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if access[edge] == size:
                    dereference(boilerplate).push_back(edge)


def mark_dag_parents(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray heads,
    ndarray timestamps,
    ndarray ownership,
    slay_hydra: bool = True,
    alloc_capsule=None,
) -> np.ndarray:
    """
    :param slay_hydra: When there is a head that reaches several roots and not all of them have \
                       parents, clear the parents so that the regular check len(parents) == 0 \
                       works.
    """
    result = np.empty(len(heads), dtype=object)
    if len(hashes) == 0:
        result.fill([])
        return result
    if len(heads) == 0:
        return result
    assert heads.dtype.char == "S"
    # we cannot sort heads because the order is important
    found_heads = searchsorted_inrange(hashes, heads)
    found_heads[hashes[found_heads] != heads] = len(vertexes)
    heads = found_heads.astype(np.uint32)
    timestamps = timestamps.view(np.uint64)
    ownership = ownership.astype(np.int32, copy=False)
    cdef:
        optional[mi_heap_destroy_stl_allocator[uint32_t]] alloc
        optional[mi_vector[mi_vector[uint32_t]]] parents
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        const uint64_t[:] timestamps_view = timestamps
        const int32_t[:] ownership_view = ownership
        bool slay_hydra_native = slay_hydra
        Py_ssize_t len_heads = len(heads), i
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        parents.emplace(dereference(alloc))
        dereference(parents).reserve(len_heads)
        for i in range(len_heads):
            dereference(parents).emplace_back(dereference(alloc))
        full_size = _mark_dag_parents(
            vertexes_view, edges_view, heads_view, timestamps_view, ownership_view,
            slay_hydra_native, &dereference(parents))
    concat_parents = np.zeros(full_size, dtype=np.uint32)
    split_points = np.zeros(dereference(parents).size(), dtype=np.int64)
    cdef:
        uint32_t[:] concat_parents_view = concat_parents
        int64_t[:] split_points_view = split_points
    with nogil:
        _copy_parents_to_array(&dereference(parents), concat_parents_view, split_points_view)
    result = np.empty(dereference(parents).size(), dtype=object)
    result[:] = np.split(concat_parents, split_points[:-1])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _copy_parents_to_array(const mi_vector[mi_vector[uint32_t]] *parents,
                                 uint32_t[:] output,
                                 int64_t[:] splits) nogil:
    cdef:
        size_t i, offset = 0
        const mi_vector[uint32_t] *vec
    for i in range(parents.size()):
        vec = &dereference(parents)[i]  # (*parents)[i]
        memcpy(&output[offset], vec.data(), 4 * vec.size())
        offset += vec.size()
        splits[i] = offset


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int64_t _mark_dag_parents(const uint32_t[:] vertexes,
                               const uint32_t[:] edges,
                               const uint32_t[:] heads,
                               const uint64_t[:] timestamps,
                               const int32_t[:] ownership,
                               bool slay_hydra,
                               mi_vector[mi_vector[uint32_t]] *parents) nogil:
    cdef:
        uint32_t not_found = len(vertexes), head, peek, edge, peek_owner, parent, beg, end
        uint64_t timestamp, head_timestamp
        int64_t i, j, sum_len = 0
        size_t p
        bool reached_root
        optional[mi_vector[char]] visited
        optional[mi_vector[uint32_t]] boilerplate
        mi_vector[uint32_t] *my_parents
    visited.emplace(len(vertexes) - 1, parents.get_allocator())
    boilerplate.emplace(parents.get_allocator())
    for i in range(len(heads)):
        head = heads[i]
        if head == not_found:
            continue
        head_timestamp = timestamps[i]
        my_parents = &dereference(parents)[i]
        reached_root = False
        memset(dereference(visited).data(), 0, dereference(visited).size())
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if dereference(visited)[peek]:
                continue
            dereference(visited)[peek] = 1
            peek_owner = ownership[peek]
            if peek_owner != i:
                timestamp = timestamps[peek_owner]
                if timestamp < head_timestamp:
                    # we don't expect many parents so scan linear
                    for p in range(my_parents.size()):
                        parent = dereference(my_parents)[p]
                        if parent == peek_owner:
                            break
                        if timestamp > timestamps[parent]:
                            sum_len += 1
                            my_parents.insert(my_parents.begin() + p, peek_owner)
                            break
                    else:
                        sum_len += 1
                        my_parents.push_back(peek_owner)
                    continue
            beg, end = vertexes[peek], vertexes[peek + 1]
            if beg == end:
                reached_root = True
            for j in range(beg, end):
                edge = edges[j]
                if not dereference(visited)[edge]:
                    dereference(boilerplate).push_back(edge)
        if reached_root and slay_hydra:
            # case when there are several different histories merged together
            my_parents.clear()
    return sum_len


def extract_first_parents(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray heads,
    max_depth: long = 0,
) -> np.ndarray:
    assert heads.dtype.char == "S"
    heads = np.sort(heads)
    if len(hashes):
        found_heads = searchsorted_inrange(hashes, heads)
        heads = found_heads[hashes[found_heads] == heads].astype(np.uint32)
    else:
        heads = np.array([], dtype=np.uint32)
    first_parents = np.zeros_like(hashes, dtype=np.bool_)
    cdef:
        long max_depth_native = max_depth
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        char[:] first_parents_view = first_parents
    with nogil:
        _extract_first_parents(
            vertexes_view, edges_view, heads_view, max_depth_native, first_parents_view,
        )
    return hashes[first_parents]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_first_parents(const uint32_t[:] vertexes,
                                 const uint32_t[:] edges,
                                 const uint32_t[:] heads,
                                 long max_depth,
                                 char[:] first_parents) nogil:
    cdef:
        uint32_t head
        long i, depth
    for i in range(len(heads)):
        head = heads[i]
        depth = 0
        while not first_parents[head]:
            first_parents[head] = 1
            depth += 1
            if max_depth > 0 and depth >= max_depth:
                break
            if vertexes[head + 1] > vertexes[head]:
                head = edges[vertexes[head]]
            else:
                break


def partition_dag(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray seeds,
    alloc_capsule=None,
) -> np.ndarray:
    seeds = np.sort(seeds)
    if len(hashes):
        found_seeds = searchsorted_inrange(hashes, seeds)
        seeds = found_seeds[hashes[found_seeds] == seeds].astype(np.uint32)
    else:
        seeds = np.array([], dtype=np.uint32)
    borders = np.zeros_like(hashes, dtype=np.bool_)
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] seeds_view = seeds
        char[:] borders_view = borders
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        _partition_dag(vertexes_view, edges_view, seeds_view, borders_view, dereference(alloc))
    return hashes[borders]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _partition_dag(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] heads,
    char[:] borders,
    mi_heap_destroy_stl_allocator[char] &alloc,
) nogil:
    cdef:
        optional[mi_vector[uint32_t]] boilerplate
        optional[mi_vector[char]] visited_alloc
        char *visited
        long i, v
        uint32_t head, edge, peek, j
    boilerplate.emplace(alloc)
    visited_alloc.emplace(len(vertexes) - 1, alloc)
    visited = dereference(visited_alloc).data()
    for i in range(len(heads)):
        head = heads[i]
        # traverse the DAG from top to bottom, marking the visited nodes
        memset(visited, 0, dereference(visited_alloc).size())
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if visited[peek]:
                continue
            visited[peek] = 1
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not visited[edge]:
                    dereference(boilerplate).push_back(edge)
        # include every visited node with back edges from non-visited nodes in the partition_dag
        for v in range(len(vertexes) - 1):
            if visited[v]:
                continue
            for j in range(vertexes[v], vertexes[v + 1]):
                edge = edges[j]
                if visited[edge]:
                    borders[edge] = 1


def extract_pr_commits(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray pr_merges,
    alloc_capsule=None,
) -> Sequence[np.ndarray]:
    if len(hashes) == 0:
        return [np.array([], dtype="S40") for _ in pr_merges]
    order = np.argsort(pr_merges)
    pr_merges = pr_merges[order]
    found_pr_merges = searchsorted_inrange(hashes, pr_merges)
    found_pr_merges[hashes[found_pr_merges] != pr_merges] = len(vertexes)
    pr_merges = found_pr_merges.astype(np.uint32)[np.argsort(order)]
    left_vertexes_map = np.zeros(len(hashes), dtype=np.int8)
    cdef:
        optional[mi_vector[mi_vector[uint32_t]]] pr_commits
        optional[mi_heap_destroy_stl_allocator[uint32_t]] alloc
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] pr_merges_view = pr_merges
        int8_t[:] left_vertexes_map_view = left_vertexes_map
        size_t pr_merges_len = len(pr_merges), i, j, v
        char *hashes_data = PyArray_BYTES(hashes)
        mi_vector[uint32_t] *pr_vertexes
        ndarray pr_hashes
        char *pr_hashes_data
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        pr_commits.emplace(dereference(alloc))
        dereference(pr_commits).reserve(pr_merges_len)
        for i in range(pr_merges_len):
            dereference(pr_commits).emplace_back(dereference(alloc))
        _extract_pr_commits(
            vertexes_view,
            edges_view,
            pr_merges_view,
            left_vertexes_map_view,
            dereference(pr_commits),
        )
    result = np.zeros(pr_merges_len, dtype=object)
    for i in range(pr_merges_len):
        pr_vertexes = &dereference(pr_commits)[i]
        pr_hashes = np.empty(pr_vertexes.size(), dtype=hashes.dtype)
        pr_hashes_data = PyArray_BYTES(pr_hashes)
        for j in range(pr_vertexes.size()):
            v = dereference(pr_vertexes)[j]
            memcpy(pr_hashes_data + j * 40, hashes_data + v * 40, 40)
        result[i] = pr_hashes
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_pr_commits(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] pr_merges,
    int8_t[:] left_vertexes_map,
    mi_vector[mi_vector[uint32_t]] &pr_commits,
) nogil:
    cdef:
        long i
        uint32_t first, last, v, j, edge, peek
        uint32_t oob = len(vertexes)
        mi_vector[uint32_t] *my_pr_commits
        optional[mi_vector[uint32_t]] boilerplate
    boilerplate.emplace(pr_commits.get_allocator())
    dereference(boilerplate).reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(pr_merges)):
        v = pr_merges[i]
        if v == oob:
            continue
        first = vertexes[v]
        last = vertexes[v + 1]
        if last - first != 2:  # we don't support octopus
            continue

        # extract the full sub-DAG of the main branch
        left_vertexes_map[:] = 0
        dereference(boilerplate).clear()
        dereference(boilerplate).push_back(edges[first])
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    dereference(boilerplate).push_back(edge)

        # traverse the DAG starting from the side edge, stop on any vertex in the main sub-DAG
        my_pr_commits = &pr_commits[i]
        my_pr_commits.push_back(v)  # include the merge commit in the PR
        dereference(boilerplate).push_back(edges[last - 1])
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            my_pr_commits.push_back(peek)
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    dereference(boilerplate).push_back(edge)


def extract_independent_ownership(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray heads,
    ndarray stops,
    alloc_capsule=None,
) -> np.ndarray:
    if len(hashes) == 0 or len(heads) == 0:
        result = np.empty(len(heads), dtype=object)
        result.fill(np.array([], dtype="S40"))
        return result
    assert heads.dtype.char == "S"
    assert len(heads) == len(stops)
    # we cannot sort heads because the order is important
    found_heads = searchsorted_inrange(hashes, heads)
    found_heads[hashes[found_heads] != heads] = len(vertexes)
    heads = found_heads.astype(np.uint32)
    del found_heads
    all_stops = np.concatenate(stops)
    found_stops = searchsorted_inrange(hashes, all_stops)
    found_stops[hashes[found_stops] != all_stops] = len(vertexes)
    splits = np.zeros(len(stops) + 1, dtype=np.int64)
    np.cumsum([len(arr) for arr in stops], out=splits[1:])
    stops = found_stops.astype(np.uint32)
    left_vertexes_map = np.zeros_like(vertexes)
    left_vertexes = left_edges = np.array([], dtype=np.uint32)
    single_slot = np.zeros(1, dtype=np.uint32)
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_vector[mi_vector[uint32_t]]] found_commits
        mi_vector[uint32_t] *found_commits_data
        uint32_t *head_vertexes
        size_t heads_len = len(heads)
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        const uint32_t[:] stops_view = stops
        const int64_t[:] splits_view = splits
        uint32_t[:] single_slot_view = single_slot
        uint32_t[:] left_vertexes_map_view = left_vertexes_map
        uint32_t[:] left_vertexes_view = left_vertexes
        uint32_t[:] left_edges_view = left_edges
        ndarray head_hashes
        char *head_hashes_data
        char *hashes_data = PyArray_BYTES(hashes)
        size_t i, j

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()

    with nogil:
        found_commits.emplace(dereference(alloc))
        dereference(found_commits).reserve(heads_len)
        for _ in range(heads_len):
            dereference(found_commits).emplace_back(dereference(alloc))
        _extract_independent_ownership(
            vertexes_view, edges_view, heads_view, stops_view, splits_view,
            single_slot_view, left_vertexes_map_view, left_vertexes_view, left_edges_view,
            &dereference(found_commits))
    result = np.zeros(dereference(found_commits).size(), dtype=object)
    found_commits_data = dereference(found_commits).data()
    for i in range(heads_len):
        head_hashes = np.empty(found_commits_data[i].size(), dtype="S40")
        head_hashes_data = PyArray_BYTES(head_hashes)
        head_vertexes = found_commits_data[i].data()
        for j in range(found_commits_data[i].size()):
            memcpy(head_hashes_data + j * 40, hashes_data + (<size_t>head_vertexes[j]) * 40, 40)
        result[i] = head_hashes
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_independent_ownership(
    const uint32_t[:] vertexes,
    const uint32_t[:] edges,
    const uint32_t[:] heads,
    const uint32_t[:] stops,
    const int64_t[:] splits,
    uint32_t[:] single_slot,
    uint32_t[:] left_vertexes_map,
    uint32_t[:] left_vertexes,
    uint32_t[:] left_edges,
    mi_vector[mi_vector[uint32_t]] *result,
) nogil:
    cdef:
        int64_t i, p
        uint32_t j, head, parent, count, peek, edge
        uint32_t oob = len(vertexes)
        mi_vector[uint32_t] *head_result
        optional[mi_vector[uint32_t]] boilerplate
        bool has_parent
    boilerplate.emplace(result.get_allocator())
    dereference(boilerplate).reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(heads)):
        head = heads[i]
        if head == oob:
            continue
        head_result = &dereference(result)[i]
        left_vertexes_map[:] = 0
        has_parent = False
        for p in range(splits[i], splits[i + 1]):
            parent = stops[p]
            if parent == oob:
                continue
            has_parent = True
            single_slot[0] = parent
            _extract_subdag(
                vertexes,
                edges,
                single_slot,
                True,
                left_vertexes_map,
                left_vertexes,
                left_edges,
                <mi_heap_destroy_stl_allocator[char]> result.get_allocator(),
            )
        if not has_parent:
            single_slot[0] = head
            count = _extract_subdag(
                vertexes,
                edges,
                single_slot,
                False,
                left_vertexes_map,
                left_vertexes,
                left_edges,
                <mi_heap_destroy_stl_allocator[char]> result.get_allocator(),
            )
            head_result.reserve(count)
            for j in range(count):
                head_result.push_back(left_vertexes_map[j])
            continue
        dereference(boilerplate).push_back(head)
        while not dereference(boilerplate).empty():
            peek = dereference(boilerplate).back()
            dereference(boilerplate).pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            head_result.push_back(peek)
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    dereference(boilerplate).push_back(edge)


def lookup_children(
    ndarray hashes,
    ndarray vertexes,
    ndarray edges,
    ndarray parents,
    ndarray output,
    ndarray output_indexes,
) -> None:
    cdef:
        long[:] parent_vertexes
        uint32_t[:] vertexes_arr = vertexes
        uint32_t[:] edges_arr = edges
        long[:] output_indexes_arr = output_indexes
        long i
        uint32_t vertex, edge_begin, edge_end, edge
        char *hashes_data = PyArray_BYTES(hashes)
        PyObject *children
        PyObject **output_data = <PyObject **> PyArray_BYTES(output)
    parent_vertexes = np.searchsorted(hashes, parents)
    for i in range(len(parents)):
        vertex = parent_vertexes[i]
        edge_begin = vertexes_arr[vertex]
        edge_end = vertexes_arr[vertex + 1]
        if edge_begin == edge_end:
            continue
        children = PyList_New(edge_end - edge_begin)
        for edge in range(edge_begin, edge_end):
            PyList_SET_ITEM(
                children,
                edge - edge_begin,
                PyBytes_FromStringAndSize(hashes_data + edges_arr[edge] * 40, 40),
            )
        output_data[output_indexes_arr[i]] = children


def compose_sha_values(ndarray shas not None, str suffix) -> str:
    assert shas.dtype == "S40"
    assert PyArray_NDIM(shas) == 1
    assert PyArray_IS_C_CONTIGUOUS(shas)
    cdef:
        npy_intp count = PyArray_DIM(shas, 0), i
        Py_ssize_t suffix_len = len(suffix)
        str result = PyUnicode_New(count * (40 + 2 + 2 + 1) - 1 + 1 + 6 + 1 + 1 + suffix_len, 127)
        char *rbuf = <char *> PyUnicode_DATA(<PyObject *> result)
        char *shas_data = PyArray_BYTES(shas)
    with nogil:
        memcpy(rbuf, b"(VALUES ", 8)
        rbuf += 8
        for i in range(count):
            rbuf[0] = ord(b"(")
            rbuf[1] = ord(b"'")
            memcpy(rbuf + 2, shas_data + i * 40, 40)
            rbuf[42] = ord(b"'")
            rbuf[43] = ord(b")")
            rbuf[44] = ord(b",")
            rbuf += 45
        rbuf[-1] = ord(b")")
        memcpy(rbuf, PyUnicode_DATA(<PyObject *> suffix), suffix_len)
    return result
