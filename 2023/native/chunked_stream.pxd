from athenian.api.native.mi_heap_destroy_stl_allocator cimport mi_heap_destroy_stl_allocator


cdef extern from "chunked_stream.h" nogil:
    cdef cppclass chunked_stream[I=*]:
        chunked_stream chunked_stream[X](mi_heap_destroy_stl_allocator[X] &) except +
        void write(const void *buffer, size_t size)
        size_t dump(char *output, size_t output_size)
        size_t size()
