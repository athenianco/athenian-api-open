cdef extern from "<string_view>" namespace "std" nogil:
    cppclass string_view:
        string_view() except +
        string_view(const char *, size_t) except +
        const char *data()
        size_t size()
