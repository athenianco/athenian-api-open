# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

import cython

from cpython cimport PyObject
from cython.operator cimport dereference as deref
from libc.stddef cimport wchar_t
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memchr, memcpy
from numpy cimport (
    PyArray_DATA,
    PyArray_DESCR,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    PyArray_STRIDE,
    dtype as np_dtype,
    ndarray,
)

from athenian.api.internal.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.native.cpython cimport (
    Py_INCREF,
    Py_None,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
)
from athenian.api.native.mi_heap_destroy_stl_allocator cimport (
    mi_heap_destroy_stl_allocator,
    mi_unordered_map,
    mi_unordered_set,
    pair,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.string_view cimport string_view

import numpy as np


cdef extern from "wchar.h" nogil:
    wchar_t *wmemchr(const wchar_t *, wchar_t, size_t)


def unordered_unique(ndarray arr not None) -> np.ndarray:
    cdef:
        np_dtype dtype = <np_dtype>PyArray_DESCR(arr)
    assert PyArray_NDIM(arr) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr)
    if dtype.kind == b"S" or dtype.kind == b"U":
        return _unordered_unique_str(arr, dtype)
    elif dtype.kind == b"i" or dtype.kind == b"u":
        if dtype.itemsize == 8:
            return _unordered_unique_int[int64_t](arr, dtype, 0)
        elif dtype.itemsize == 4:
            return _unordered_unique_int[int64_t](arr, dtype, 4)
        else:
            raise AssertionError(f"dtype {dtype} is not supported")
    elif dtype.kind == b"O":
        return _unordered_unique_pystr(arr)
    else:
        raise AssertionError(f"dtype {dtype} is not supported")


@cython.cdivision(True)
cdef ndarray _unordered_unique_pystr(ndarray arr):
    cdef:
        PyObject **data_in = <PyObject **>PyArray_DATA(arr)
        PyObject **data_out
        PyObject *str_obj
        char *str_data
        unsigned int str_kind
        Py_ssize_t str_len
        int64_t i, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0) >> 3
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_map[string_view, int64_t]] hashtable
        pair[string_view, int64_t] it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            str_obj = data_in[i * stride]
            if str_obj == Py_None:
                continue
            str_data = <char *> PyUnicode_DATA(str_obj)
            str_len = PyUnicode_GET_LENGTH(str_obj)
            str_kind = PyUnicode_KIND(str_obj)
            deref(hashtable).try_emplace(string_view(str_data, str_len * str_kind), i)

    result = np.empty(deref(hashtable).size(), dtype=object)
    data_out = <PyObject **>PyArray_DATA(result)
    i = 0
    for it in deref(hashtable):
        str_obj = data_in[it.second]
        data_out[i] = str_obj
        Py_INCREF(str_obj)
        i += 1
    return result


@cython.cdivision(True)
cdef ndarray _unordered_unique_str(ndarray arr, np_dtype dtype):
    cdef:
        char *data = <char *>PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        optional[mi_heap_destroy_stl_allocator[string_view]] alloc
        optional[mi_unordered_set[string_view]] hashtable
        string_view it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            deref(hashtable).emplace(data + i * stride, itemsize)

    result = np.empty(deref(hashtable).size(), dtype=dtype)

    with nogil:
        data = <char *>PyArray_DATA(result)
        i = 0
        for it in deref(hashtable):
            memcpy(data + i * itemsize, it.data(), itemsize)
            i += 1
    return result


ctypedef fused varint:
    int64_t
    int32_t


@cython.cdivision(True)
cdef ndarray _unordered_unique_int(ndarray arr, np_dtype dtype, varint _):
    cdef:
        char *data = <char *> PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        optional[mi_heap_destroy_stl_allocator[varint]] alloc
        optional[mi_unordered_set[varint]] hashtable
        varint it
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length // 16)
        for i in range(length):
            deref(hashtable).emplace((<varint *>(data + i * stride))[0])

    result = np.empty(deref(hashtable).size(), dtype=dtype)

    with nogil:
        data = <char *> PyArray_DATA(result)
        i = 0
        for it in deref(hashtable):
            (<varint *>(data + i * itemsize))[0] = it
            i += 1
    return result


def in1d_str(
    ndarray trial not None,
    ndarray dictionary not None,
    bint skip_leading_zeros = False,
) -> np.ndarray:
    cdef:
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
    assert PyArray_NDIM(trial) == 1
    assert PyArray_NDIM(dictionary) == 1
    assert dtype_trial.kind == b"S" or dtype_trial.kind == b"U"
    assert dtype_trial.kind == dtype_dict.kind
    return _in1d_str(trial, dictionary, dtype_trial.kind == b"S", skip_leading_zeros)


cdef ndarray _in1d_str(ndarray trial, ndarray dictionary, bint is_char, int skip_leading_zeros):
    cdef:
        char *data_trial = <char *>PyArray_DATA(trial)
        char *data_dictionary = <char *> PyArray_DATA(dictionary)
        char *output
        char *s
        char *nullptr
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
        int64_t i, size, \
            itemsize = dtype_dict.itemsize, \
            length = PyArray_DIM(dictionary, 0), \
            stride = PyArray_STRIDE(dictionary, 0)
        optional[mi_heap_destroy_stl_allocator[string_view]] alloc
        optional[mi_unordered_set[string_view]] hashtable
        mi_unordered_set[string_view].iterator end
        ndarray result

    with nogil:
        alloc.emplace()
        hashtable.emplace(deref(alloc))
        deref(hashtable).reserve(length * 4)
        if is_char:
            for i in range(length):
                s = data_dictionary + i * stride
                nullptr = s
                if skip_leading_zeros:
                    while nullptr < (s + itemsize) and nullptr[0] == 0:
                        nullptr += 1
                nullptr = <char *> memchr(nullptr, 0, itemsize + (s - nullptr))
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                deref(hashtable).emplace(s, size)
        else:
            for i in range(length):
                s = data_dictionary + i * stride
                nullptr = <char *> wmemchr(<wchar_t *>s, 0, itemsize >> 2)
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                deref(hashtable).emplace(s, size)

        itemsize = dtype_trial.itemsize
        length = PyArray_DIM(trial, 0)
        stride = PyArray_STRIDE(trial, 0)

    result = np.empty(length, dtype=bool)

    with nogil:
        output = <char *>PyArray_DATA(result)
        end = deref(hashtable).end()
        if is_char:
            for i in range(length):
                s = data_trial + i * stride
                nullptr = s
                if skip_leading_zeros:
                    while nullptr < (s + itemsize) and nullptr[0] == 0:
                        nullptr += 1
                nullptr = <char *> memchr(nullptr, 0, itemsize + (s - nullptr))
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                output[i] = deref(hashtable).find(string_view(s, size)) != end
        else:
            for i in range(length):
                s = data_trial + i * stride
                nullptr = <char *> wmemchr(<wchar_t *> s, 0, itemsize >> 2)
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                output[i] = deref(hashtable).find(string_view(s, size)) != end
    return result


def map_array_values(
    ndarray arr not None,
    ndarray map_keys not None,
    ndarray map_values not None,
    miss_value,
) -> np.ndarray:
    """Map the values in the array `arr` using the dictionary expressed by two arrays.

    `map_keys` and `map_values` must have the same length and together represent the translation
    array.
    `map_keys` must be sorted.
    Values in `arr` not found in `map_keys` will be mapped to `miss_value`, which datatype
    should be compatible with `map_values` datatype.
    """
    cdef:
        ndarray found_keys_indexes, mapped, non_matching_keys

    assert len(map_keys) == len(map_values)
    if len(map_keys) == 0:
        return np.full(len(arr), miss_value)
    # indexes selecting from map_key in the same order as ar
    found_keys_indexes = searchsorted_inrange(map_keys, arr)
    mapped = map_values[found_keys_indexes]

    # found_keys_indexes will also have an index for ar elements not present in map_keys
    # these positions must be set to miss_value
    non_matching_keys = map_keys[found_keys_indexes] != arr
    mapped[non_matching_keys] = miss_value
    return mapped
