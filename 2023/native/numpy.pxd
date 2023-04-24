from cpython cimport PyObject
from numpy cimport dtype as npdtype, npy_int64, npy_intp

from athenian.api.native.cpython cimport PyTypeObject


cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyArray_Type
    PyTypeObject PyDatetimeArrType_Type
    PyTypeObject PyDoubleArrType_Type
    PyTypeObject PyIntegerArrType_Type
    PyTypeObject PyFloatArrType_Type
    PyTypeObject PyTimedeltaArrType_Type

    enum: NPY_DATETIME_NAT

    ctypedef struct PyArray_Descr:
        char kind
        char type
        char byteorder
        char flags
        int type_num
        int itemsize "elsize"
        int alignment

    PyObject *PyArray_NewFromDescr(
        PyTypeObject *subtype,
        PyArray_Descr *descr,
        int nd,
        const npy_intp *dims,
        const npy_intp *strides,
        void *data,
        int flags,
        PyObject *obj,
    )
    npdtype PyArray_DescrNew(npdtype)

    void *PyArray_DATA(PyObject *) nogil
    char *PyArray_BYTES(PyObject *) nogil
    npy_intp PyArray_DIM(PyObject *, size_t) nogil
    npy_intp PyArray_STRIDE(PyObject *, size_t) nogil
    int PyArray_NDIM(PyObject *) nogil
    npy_intp PyArray_ITEMSIZE(PyObject *) nogil
    bint PyArray_CheckExact(PyObject *) nogil
    PyArray_Descr *PyArray_DESCR(PyObject *) nogil
    int PyArray_TYPE(PyObject *) nogil
    bint PyArray_IS_C_CONTIGUOUS(PyObject *) nogil
    bint PyArray_IS_F_CONTIGUOUS(PyObject *) nogil
    void PyArray_ScalarAsCtype(PyObject *scalar, void *ctypeptr) nogil

    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_ERROR = -1
        NPY_FR_M = 1
        NPY_FR_W = 2
        NPY_FR_D = 4
        NPY_FR_h = 5
        NPY_FR_m = 6
        NPY_FR_s = 7
        NPY_FR_ms = 8
        NPY_FR_us = 9
        NPY_FR_ns = 10
        NPY_FR_ps = 11
        NPY_FR_fs = 12
        NPY_FR_as = 13
        NPY_FR_GENERIC = 14

    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int num

    ctypedef struct PyDatetimeScalarObject:
        npy_int64 obval
        PyArray_DatetimeMetaData obmeta
