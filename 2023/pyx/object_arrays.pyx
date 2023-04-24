# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib
# distutils: extra_compile_args = -std=c++17 -mavx2 -ftree-vectorize

cimport cython
from cpython cimport Py_INCREF, PyObject
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_Check
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_Check
from cpython.memoryview cimport PyMemoryView_Check, PyMemoryView_GET_BUFFER
from cpython.unicode cimport PyUnicode_Check
from cython.operator cimport dereference
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memcpy, memset
from numpy cimport (
    NPY_ARRAY_C_CONTIGUOUS,
    NPY_OBJECT,
    PyArray_CheckExact,
    PyArray_DATA,
    PyArray_Descr,
    PyArray_DescrFromType,
    PyArray_DIM,
    PyArray_GETCONTIGUOUS,
    PyArray_ISOBJECT,
    PyArray_ISSTRING,
    PyArray_NDIM,
    PyArray_SetBaseObject,
    dtype as npdtype,
    import_array,
    ndarray,
    npy_bool,
    npy_intp,
)

from athenian.api.native.chunked_stream cimport chunked_stream
from athenian.api.native.cpython cimport (
    Py_None,
    Py_True,
    PyBytes_GET_SIZE,
    PyList_CheckExact,
    PyList_GET_ITEM,
    PyList_GET_SIZE,
    PyTuple_GET_ITEM,
    PyTypeObject,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
)
from athenian.api.native.mi_heap_destroy_stl_allocator cimport mi_heap_destroy_stl_allocator
from athenian.api.native.numpy cimport (
    PyArray_DESCR,
    PyArray_DescrNew,
    PyArray_NewFromDescr,
    PyArray_Type,
)
from athenian.api.native.optional cimport optional

import asyncpg
import numpy as np

import_array()


cdef extern from "asyncpg_recordobj.h":
    PyObject *ApgRecord_GET_ITEM(PyObject *, int)


@cython.boundscheck(False)
def to_object_arrays(list rows not None, int columns) -> np.ndarray:
    """
    Convert a list of tuples or asyncpg.Record-s into an object array. Any subclass of
    tuple in `rows` will be casted to tuple.

    Parameters
    ----------
    rows: 2-d array (N, K)
        list of tuples to be converted into an array. Each tuple must be of equal length,
        otherwise, the results are undefined.
    columns: number of columns in each row.

    Returns
    -------
    np.ndarray[object, ndim=2]
    """
    cdef:
        Py_ssize_t i, j, size
        ndarray[object, ndim=2] result
        PyObject *record

    size = len(rows)

    result = np.empty((columns, size), dtype=object)
    if size == 0:
        return result

    if isinstance(rows[0], asyncpg.Record):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(columns):
                result[j, i] = <object>ApgRecord_GET_ITEM(record, j)
    elif isinstance(rows[0], tuple):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(columns):
                result[j, i] = <object>PyTuple_GET_ITEM(record, j)
    else:
        # convert to tuple
        for i in range(size):
            row = tuple(rows[i])
            for j in range(columns):
                result[j, i] = row[j]

    return result


def as_bool(ndarray arr not None) -> np.ndarray:
    if arr.dtype == bool:
        return arr
    assert arr.dtype == object
    assert arr.ndim == 1
    new_arr = np.empty(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _as_bool_vec(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _as_bool_vec(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        # Py_None and Py_False become 0
        out_arr[i] = Py_True == (<const PyObject **> (obj_arr + i * stride))[0]



def is_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.zeros(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_null_vec(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_null_vec(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None == (<const PyObject **> (obj_arr + i * stride))[0]


def is_not_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.ones(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_not_null(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_not_null(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None != (<const PyObject **> (obj_arr + i * stride))[0]


def nested_lengths(arr not None, output=None) -> np.ndarray:
    cdef:
        long size
        bint is_array = PyArray_CheckExact(arr)
        ndarray result

    if is_array:
        assert PyArray_ISOBJECT(arr) or PyArray_ISSTRING(arr)
        assert PyArray_NDIM(arr) == 1
        size = PyArray_DIM(arr, 0)
    else:
        assert PyList_CheckExact(<PyObject *> arr)
        size = PyList_GET_SIZE(<PyObject *> arr)

    if output is None:
        result = np.zeros(size, dtype=int)
    else:
        assert PyArray_CheckExact(output)
        assert output.dtype == int
        result = output
    if size == 0:
        return result
    if is_array:
        return _nested_lengths_arr(arr, size, result)
    return _nested_lengths_list(<PyObject *> arr, size, result)


cdef ndarray _nested_lengths_arr(ndarray arr, long size, ndarray result):
    cdef:
        PyObject **elements = <PyObject **>PyArray_DATA(arr)
        PyObject *element
        long i
        long *result_data

    if PyArray_ISSTRING(arr):
        return np.char.str_len(arr)

    result_data = <long *>PyArray_DATA(result)
    element = elements[0]
    if PyArray_CheckExact(<object> element):
        for i in range(size):
            result_data[i] = PyArray_DIM(<ndarray> elements[i], 0)
    elif PyList_CheckExact(element):
        for i in range(size):
            result_data[i] = PyList_GET_SIZE(elements[i])
    elif PyUnicode_Check(<object> element):
        for i in range(size):
            result_data[i] = PyUnicode_GET_LENGTH(elements[i])
    elif PyBytes_Check(<object> element):
        for i in range(size):
            result_data[i] = PyBytes_GET_SIZE(elements[i])
    else:
        raise AssertionError(f"Unsupported nested type: {type(<object> element).__name__}")
    return result


cdef ndarray _nested_lengths_list(PyObject *arr, long size, ndarray result):
    cdef:
        PyObject *element
        long i
        long *result_data

    result_data = <long *>PyArray_DATA(result)
    element = PyList_GET_ITEM(arr, 0)
    if PyArray_CheckExact(<object> element):
        for i in range(size):
            result_data[i] = PyArray_DIM(<ndarray> PyList_GET_ITEM(arr, i), 0)
    elif PyList_CheckExact(element):
        for i in range(size):
            result_data[i] = PyList_GET_SIZE(PyList_GET_ITEM(arr, i))
    elif PyUnicode_Check(<object> element):
        for i in range(size):
            result_data[i] = PyUnicode_GET_LENGTH(PyList_GET_ITEM(arr, i))
    elif PyBytes_Check(<object> element):
        for i in range(size):
            result_data[i] = PyBytes_GET_SIZE(PyList_GET_ITEM(arr, i))
    else:
        raise AssertionError(f"Unsupported nested type: {type(<object> element).__name__}")
    return result


def array_from_buffer(buffer not None, npdtype dtype, npy_intp count, npy_intp offset=0) -> ndarray:
    cdef:
        void *data
        ndarray arr
    if PyBytes_Check(buffer):
        data = PyBytes_AS_STRING(buffer) + offset
    elif PyByteArray_Check(buffer):
        data = PyByteArray_AS_STRING(buffer) + offset
    elif PyMemoryView_Check(buffer):
        data = (<char *>PyMemoryView_GET_BUFFER(buffer).buf) + offset
    else:
        raise ValueError(f"Unsupported buffer type: {type(buffer).__name__}")
    Py_INCREF(dtype)
    Py_INCREF(buffer)
    arr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &count,
        NULL,
        data,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    PyArray_SetBaseObject(arr, buffer)
    return arr


def array_of_objects(int length, fill_value) -> ndarray:
    cdef:
        ndarray arr
        npdtype objdtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_OBJECT))
        npy_intp nplength = length, i
        PyObject **data
        PyObject *obj = <PyObject *> fill_value

    arr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &nplength,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(objdtype)
    data = <PyObject **> PyArray_DATA(arr)
    for i in range(nplength):
        data[i] = obj
    obj.ob_refcnt += nplength
    return arr


def vectorize_numpy_struct_scalar_field(cls, structs, npdtype dtype, long offset) -> np.ndarray:
    cdef:
        ndarray arr
        npy_intp length = len(structs), i = 0
        npy_intp itemsize = dtype.itemsize
        char *arr_data
        Py_ssize_t struct_data_offset = (<PyTypeObject *> cls).tp_members[1].offset
        PyObject *struct_data_obj
        char *struct_data
    arr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &length,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(dtype)
    arr_data = <char *> PyArray_DATA(arr)

    for struct in structs:
        struct_data_obj = dereference(
            <PyObject **> ((<char *> <PyObject *> struct) + struct_data_offset)
        )
        if PyBytes_Check(<object> struct_data_obj):
            struct_data = PyBytes_AS_STRING(<object> struct_data_obj)
        elif PyMemoryView_Check(<object> struct_data_obj):
            struct_data = <char *>PyMemoryView_GET_BUFFER(<object> struct_data_obj).buf
        elif PyByteArray_Check(<object> struct_data_obj):
            struct_data = PyByteArray_AS_STRING(<object> struct_data_obj)
        else:
            raise AssertionError(f"unsupported buffer type: {type(struct.data)}")
        memcpy(arr_data + i * itemsize, struct_data + offset, itemsize)
        i += 1
    return arr


def vectorize_numpy_struct_array_field(
    cls,
    structs,
    npdtype dtype,
    long offset,
) -> tuple[np.ndarray, np.ndarray]:
    cdef:
        ndarray arr, arr_offsets
        npy_intp length = len(structs) + 1, i = 0
        npy_intp itemsize = dtype.itemsize
        char *arr_data
        Py_ssize_t struct_data_offset = (<PyTypeObject *> cls).tp_members[1].offset
        PyObject *struct_data_obj
        char *struct_data
        int32_t field_offset, field_count
        int32_t *field
        npdtype int_dtype = npdtype(int)
        int64_t pos = 0, delta
        optional[chunked_stream] dump
        optional[mi_heap_destroy_stl_allocator[char]] alloc

    alloc.emplace()
    dump.emplace(dereference(alloc))
    arr_offsets = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> int_dtype,
        1,
        &length,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(int_dtype)
    offsets_data = <int64_t *> PyArray_DATA(arr_offsets)

    length = 0
    for struct in structs:
        struct_data_obj = dereference(
            <PyObject **> ((<char *> <PyObject *> struct) + struct_data_offset)
        )
        if PyBytes_Check(<object> struct_data_obj):
            struct_data = PyBytes_AS_STRING(<object> struct_data_obj)
        elif PyMemoryView_Check(<object> struct_data_obj):
            struct_data = <char *>PyMemoryView_GET_BUFFER(<object> struct_data_obj).buf
        elif PyByteArray_Check(<object> struct_data_obj):
            struct_data = PyByteArray_AS_STRING(<object> struct_data_obj)
        else:
            raise AssertionError(f"unsupported buffer type: {type(struct.data)}")
        field = <int32_t *>(struct_data + offset)
        field_offset = field[0]
        field_count = field[1]
        offsets_data[i] = length
        length += field_count
        delta = itemsize * field_count
        dereference(dump).write(struct_data + field_offset, delta)
        pos += delta
        i += 1
    offsets_data[i] = length

    arr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &length,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(dtype)
    dereference(dump).dump(<char *>PyArray_DATA(arr), pos)

    return arr, arr_offsets


def objects_to_pyunicode_bytes(ndarray arr not None, char_limit=None) -> ndarray:
    assert PyArray_NDIM(arr) == 1
    assert PyArray_DESCR(<PyObject *> arr).kind == b"O"

    cdef npy_intp length = PyArray_DIM(arr, 0)

    if length == 0:
        return np.array([], dtype="S")

    arr = PyArray_GETCONTIGUOUS(arr)

    cdef:
        npy_intp i, max_itemsize = 0
        PyObject **data = <PyObject **> PyArray_DATA(arr)
        PyObject *obj
        Py_ssize_t i_itemsize, char_limit_native
        npdtype dtype
        ndarray converted
        char *converted_data
        char *head

    if char_limit is None:
        for i in range(length):
            obj = data[i]
            if obj == Py_None:
                i_itemsize = 4
            else:
                assert PyUnicode_Check(<object> obj), f"arr[{i}]: {arr[i]}"
                i_itemsize = PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj)
            if i_itemsize > max_itemsize:
                max_itemsize = i_itemsize
    else:
        assert char_limit > 0
        char_limit_native = char_limit
        for i in range(length):
            obj = data[i]
            if obj == Py_None:
                i_itemsize = 4
            else:
                assert PyUnicode_Check(<object> obj), f"arr[{i}]: {arr[i]}"
                i_itemsize = PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj)
            if i_itemsize >= char_limit_native:
                max_itemsize = char_limit_native
                break
            if i_itemsize > max_itemsize:
                max_itemsize = i_itemsize

    dtype = npdtype("S" + str(max_itemsize))
    converted = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &length,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(dtype)
    converted_data = <char *> PyArray_DATA(converted)

    for i in range(length):
        obj = data[i]
        head = converted_data + i * max_itemsize

        if obj == Py_None:
            i_itemsize = 4
            if i_itemsize > max_itemsize:
                i_itemsize = max_itemsize
            memcpy(head, b"None", i_itemsize)
        else:
            i_itemsize = PyUnicode_GET_LENGTH(obj) * PyUnicode_KIND(obj)
            if i_itemsize > max_itemsize:
                i_itemsize = max_itemsize
            memcpy(head, PyUnicode_DATA(obj), i_itemsize)
        memset(head + i_itemsize, 0, max_itemsize - i_itemsize)

    return converted
