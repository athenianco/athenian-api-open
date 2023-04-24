# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

from cpython.pycapsule cimport PyCapsule_New


def make_mi_heap_allocator_capsule() -> object:
    cdef mi_heap_destroy_stl_allocator[char] *alloc = new mi_heap_destroy_stl_allocator[char]()
    return PyCapsule_New(alloc, b"mi_heap_destroy_stl_allocator", _delete_mi_heap_allocator_in_capsule)
