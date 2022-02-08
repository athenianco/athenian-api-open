// no need to #include anything, this file is used internally by to_object_arrays.pyx

typedef struct {
    PyObject_VAR_HEAD

    // asyncpg specifics begin here
    // if they add another field, we will break spectacularly
    Py_hash_t self_hash;
    PyObject *desc;  // we don't care of the actual type
    PyObject *ob_item[1];  // embedded in the tail, the count matches len()
} ApgRecordObject;

#define ApgRecord_GET_ITEM(op, i) (((ApgRecordObject *)(op))->ob_item[i])
#define ApgRecord_SET_ITEM(op, i, v) (((ApgRecordObject *)(op))->ob_item[i] = v)
#define ApgRecord_GET_DESC(op) (((ApgRecordObject *)(op))->desc)
