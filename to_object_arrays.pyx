# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++

from typing import Any, List, Sequence, Tuple

import asyncpg
import numpy as np

cimport cython
from cpython cimport PyObject
from numpy cimport ndarray


cdef extern from "asyncpg_recordobj.h":
    PyObject *ApgRecord_GET_ITEM(PyObject *, int)


cdef extern from "Python.h":
    # added nogil -> from cpython cimport ...
    # these are the macros that read directly from the internal ob_items
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil


@cython.boundscheck(False)
def to_object_arrays_split(rows: List[Sequence[Any]],
                           typed_indexes: Sequence[int],
                           obj_indexes: Sequence[int],
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of tuples into an object array. Any subclass of
    tuple in `rows` will be casted to tuple.

    Parameters
    ----------
    rows : 2-d array (N, K)
        List of tuples to be converted into an array. Each tuple must be of equal length,
        otherwise, the results are undefined.
    typed_indexes : array of integers
        Sequence of integer indexes in each tuple in `rows` that select the first result.
    obj_indexes : array of integers
        Sequence of integer indexes in each tuple in `rows` that select the second result.

    Returns
    -------
    (np.ndarray[object, ndim=2], np.ndarray[object, ndim=2])
    The first array is the concatenation of columns in `rows` chosen by `typed_indexes`.
    The second array is the concatenation of columns in `rows` chosen by `object_indexes`.
    """
    cdef:
        Py_ssize_t i, j, size, cols_typed, cols_obj
        ndarray[object, ndim=2] result_typed
        ndarray[object, ndim=2] result_obj
        PyObject *record
        long[:] typed_indexes_arr
        long[:] obj_indexes_arr

    assert isinstance(rows, list)
    typed_indexes_arr = np.asarray(typed_indexes, dtype=int)
    obj_indexes_arr = np.asarray(obj_indexes, dtype=int)
    size = len(rows)
    cols_typed = len(typed_indexes_arr)
    cols_obj = len(obj_indexes_arr)

    result_typed = np.empty((cols_typed, size), dtype=object)
    result_obj = np.empty((cols_obj, size), dtype=object)
    if size == 0:
        return result_typed, result_obj

    if isinstance(rows[0], asyncpg.Record):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(cols_typed):
                result_typed[j, i] = <object>ApgRecord_GET_ITEM(record, typed_indexes_arr[j])
            for j in range(cols_obj):
                result_obj[j, i] = <object>ApgRecord_GET_ITEM(record, obj_indexes_arr[j])
    elif isinstance(rows[0], tuple):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(cols_typed):
                result_typed[j, i] = <object>PyTuple_GET_ITEM(record, typed_indexes_arr[j])
            for j in range(cols_obj):
                result_obj[j, i] = <object>PyTuple_GET_ITEM(record, obj_indexes_arr[j])
    else:
        # convert to tuple
        for i in range(size):
            row = tuple(rows[i])
            for j in range(cols_typed):
                result_typed[j, i] = row[typed_indexes_arr[j]]
            for j in range(cols_obj):
                result_obj[j, i] = row[obj_indexes_arr[j]]

    return result_typed, result_obj