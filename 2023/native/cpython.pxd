from cpython cimport PyObject

ctypedef PyObject *PyObjectPtr

cdef extern from "structmember.h":
    ctypedef struct PyMemberDef:
        const char *name
        int type
        Py_ssize_t offset
        int flags
        const char *doc

cdef extern from "Python.h":
    ctypedef PyObject *(*allocfunc)(PyTypeObject *cls, Py_ssize_t nitems)

    ctypedef struct PyTypeObject:
        allocfunc tp_alloc
        PyMemberDef *tp_members

    bint PyObject_TypeCheck(PyObject *, PyTypeObject *) nogil
    PyTypeObject *Py_TYPE(const PyObject *) nogil

    bint PyLong_CheckExact(PyObject *) nogil
    long PyLong_AsLong(PyObject *) nogil

    double PyFloat_AS_DOUBLE(PyObject *) nogil
    bint PyFloat_CheckExact(PyObject *) nogil
    bint PyFloat_Check(PyObject *) nogil

    PyObject *PyList_New(Py_ssize_t len)
    bint PyList_CheckExact(PyObject *) nogil
    Py_ssize_t PyList_GET_SIZE(PyObject *) nogil
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o) nogil

    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil

    bint PyDict_CheckExact(PyObject *) nogil
    int PyDict_Next(PyObject *p, Py_ssize_t *ppos, PyObject **pkey, PyObject **pvalue) nogil
    Py_ssize_t PyDict_Size(PyObject *p) nogil

    bint PyUnicode_Check(PyObject *) nogil
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *) nogil
    unsigned int PyUnicode_KIND(PyObject *) nogil
    void *PyUnicode_DATA(PyObject *) nogil
    Py_ssize_t PyUnicode_FindChar(
        PyObject *str, Py_UCS4 ch, Py_ssize_t start, Py_ssize_t end, int direction
    ) nogil

    bint PyBytes_Check(PyObject *) nogil
    char *PyBytes_AS_STRING(PyObject *) nogil
    Py_ssize_t PyBytes_GET_SIZE(PyObject *) nogil

    bint PyByteArray_CheckExact(PyObject *) nogil
    char *PyByteArray_AS_STRING(PyObject *) nogil

    unsigned int PyUnicode_1BYTE_KIND
    unsigned int PyUnicode_2BYTE_KIND
    unsigned int PyUnicode_4BYTE_KIND

    PyObject *Py_None
    PyObject *Py_True
    PyObject *Py_False

    PyTypeObject PyLong_Type
    PyTypeObject PyFloat_Type
    PyTypeObject PyUnicode_Type
    PyTypeObject PyBool_Type
    PyTypeObject PyList_Type
    PyTypeObject PyDict_Type
    PyTypeObject PyBaseObject_Type

    void Py_INCREF(PyObject *)
    void Py_DECREF(PyObject *)

    object PyUnicode_FromStringAndSize(const char *, Py_ssize_t)
    str PyUnicode_FromKindAndData(unsigned int kind, void *buffer, Py_ssize_t size)
    str PyUnicode_FromString(const char *)
    object PyUnicode_New(Py_ssize_t, Py_UCS4)
    PyObject *PyBytes_FromStringAndSize(char *v, Py_ssize_t len)
    PyObject *PyLong_FromLong(long v)
    PyObject *PyObject_GetItem(PyObject *o, PyObject *key)
    PyObject *PyObject_GetAttr(PyObject *o, PyObject *attr_name)

cdef extern from "datetime.h" nogil:
    bint PyDateTime_Check(PyObject *)
    bint PyDelta_Check(PyObject *)

    int PyDateTime_GET_YEAR(PyObject *)
    int PyDateTime_GET_MONTH(PyObject *)
    int PyDateTime_GET_DAY(PyObject *)

    int PyDateTime_DATE_GET_HOUR(PyObject *)
    int PyDateTime_DATE_GET_MINUTE(PyObject *)
    int PyDateTime_DATE_GET_SECOND(PyObject *)

    int PyDateTime_DELTA_GET_DAYS(PyObject *)
    int PyDateTime_DELTA_GET_SECONDS(PyObject *)

    PyObject *PyDateTime_DATE_GET_TZINFO(PyObject *)

    ctypedef struct PyDateTime_CAPI:
        PyObject *TimeZone_UTC
