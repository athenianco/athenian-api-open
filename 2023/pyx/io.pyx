# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

import pickle
from typing import Any

from cpython cimport Py_INCREF, PyBytes_FromStringAndSize, PyObject, PyTuple_New, PyTuple_SET_ITEM
from libc.stdint cimport uint32_t
from libc.string cimport memcpy

from athenian.api.native.cpython cimport PyBytes_AS_STRING

from medvedi import DataFrame


cdef extern from "<stdlib.h>" nogil:
    char *gcvt(double number, int ndigit, char *buf)


cdef extern from "<string.h>" nogil:
    size_t strnlen(const char *, size_t)


def serialize_args(tuple args, alloc_capsule=None) -> bytes:
    cdef:
        bytes result, buffer
        Py_ssize_t size = 4
        list buffers = []
        char *output
        bint is_df

    for arg in args:
        if isinstance(arg, DataFrame):
            is_df = True
            buffer = arg.serialize_unsafe()
        else:
            is_df = False
            buffer = pickle.dumps(arg)
        size += len(buffer) + 5
        buffers.append((is_df, buffer))
    result = PyBytes_FromStringAndSize(NULL, size)
    output = PyBytes_AS_STRING(<PyObject *> result)
    (<uint32_t *> output)[0] = len(buffers)
    output += 4
    for is_df, buffer in buffers:
        output[0] = is_df
        output += 1
        size = len(buffer)
        (<uint32_t *> output)[0] = size
        output += 4
        memcpy(output, PyBytes_AS_STRING(<PyObject *> buffer), size)
        output += size
    return result


def deserialize_args(bytes buffer) -> tuple[Any]:
    cdef:
        uint32_t size, i
        tuple result
        long offset = 4
        object item
        char is_df

    input = PyBytes_AS_STRING(<PyObject *> buffer)
    size = (<uint32_t *> input)[0]
    input += 4
    result = PyTuple_New(size)
    for i in range(size):
        is_df = input[0]
        input += 1
        size = (<uint32_t *> input)[0]
        input += 4
        offset += 5
        if is_df:
            item = DataFrame.deserialize_unsafe(buffer[offset: offset + size])
        else:
            item = pickle.loads(buffer[offset: offset + size])
        offset += size
        input += size
        Py_INCREF(item)
        PyTuple_SET_ITEM(result, i, item)
    return result
