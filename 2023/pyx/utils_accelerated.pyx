# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17 -mavx2

from typing import Any, Optional

from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings, default_branch_alias

from cpython cimport PyObject
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.dict cimport PyDict_GetItem
from cpython.unicode cimport PyUnicode_GET_LENGTH
from cython.operator cimport dereference as deref
from libc.string cimport memchr, memcmp, memcpy
from libcpp.memory cimport allocator, unique_ptr

from athenian.api.native.cpython cimport PyUnicode_DATA, PyUnicode_KIND
from athenian.api.native.mi_heap_destroy_stl_allocator cimport (
    empty_deleter,
    mi_heap_destroy_stl_allocator,
    mi_unordered_map,
    mi_vector,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.string_view cimport string_view


cdef extern from "string.h" nogil:
    void *memmem(
        const void *haystack, size_t haystacklen,
        const void *needle, size_t needlelen,
    )

cdef extern from "utils_accelerated.h" nogil:
    void interleave_bytes2(const char *src, size_t length, char *out)
    void interleave_bytes4(const char *src, size_t length, char *out)
    void interleave_bytes24(const char *src, size_t length, char *out)

cdef:
    mi_heap_destroy_stl_allocator[char] alloc
    optional[mi_vector[unique_ptr[char[], empty_deleter]]] _str_storage

_str_storage.emplace(alloc)

cdef:
    Py_ssize_t default_branch_alias_len = len(default_branch_alias)
    const char *default_branch_alias_data1 = <const char *> PyUnicode_DATA(
        <PyObject *> default_branch_alias
    )
    char *default_branch_alias_data2 = <char *> deref(_str_storage).emplace_back(alloc.allocate(default_branch_alias_len * 2)).get()
    char *default_branch_alias_data4 = <char *> deref(_str_storage).emplace_back(alloc.allocate(default_branch_alias_len * 4)).get()
    long branch = ReleaseMatch.branch
    long tag = ReleaseMatch.tag
    long tag_or_branch = ReleaseMatch.tag_or_branch
    long event = ReleaseMatch.event
    const char *tag_or_branch_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.tag_or_branch.name)
    Py_ssize_t tag_or_branch_name_len = len(ReleaseMatch.tag_or_branch.name)
    const char *rejected_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.rejected.name)
    Py_ssize_t rejected_name_len = len(ReleaseMatch.rejected.name)
    const char *force_push_drop_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.force_push_drop.name)
    Py_ssize_t force_push_drop_name_len = len(ReleaseMatch.force_push_drop.name)
    optional[mi_unordered_map[string_view, long]] release_match_name_to_enum

interleave_bytes2(
    default_branch_alias_data1,
    default_branch_alias_len,
    default_branch_alias_data2,
)
interleave_bytes4(
    default_branch_alias_data1,
    default_branch_alias_len,
    default_branch_alias_data4,
)

release_match_name_to_enum.emplace(alloc)
for obj in ReleaseMatch:
    deref(release_match_name_to_enum)[string_view(
        <char *>PyUnicode_DATA(<PyObject *> obj.name), PyUnicode_GET_LENGTH(obj.name)
    )] = obj
    deref(_str_storage).emplace_back(alloc.allocate(PyUnicode_GET_LENGTH(obj.name) * 2))
    interleave_bytes2(
        <char *>PyUnicode_DATA(<PyObject *> obj.name),
        PyUnicode_GET_LENGTH(obj.name),
        <char *> deref(_str_storage).back().get(),
    )
    deref(release_match_name_to_enum)[string_view(
        <char *> deref(_str_storage).back().get(), PyUnicode_GET_LENGTH(obj.name) * 2
    )] = obj
    deref(_str_storage).emplace_back(alloc.allocate(PyUnicode_GET_LENGTH(obj.name) * 4))
    interleave_bytes4(
        <char *> PyUnicode_DATA(<PyObject *> obj.name),
        PyUnicode_GET_LENGTH(obj.name),
        <char *> deref(_str_storage).back().get(),
    )
    deref(release_match_name_to_enum)[string_view(
        <char *> deref(_str_storage).back().get(), PyUnicode_GET_LENGTH(obj.name) * 4
    )] = obj


def interleave_expand(bytes src, int srckind, int dstkind):
    """
    srckind > dstkind
    dstkind=2: abcd -> a0b0c0d0 (ucs1 -> ucs2)
    srckind=1 dstkind=4: abcd -> a000b000c000d000 (ucs1 -> ucs4)
    srckind=2 dstkind=4: a0b0c0d0 -> a000b000c000d000 (ucs2 -> ucs4)
    """
    assert srckind < dstkind
    cdef:
        bytes output = PyBytes_FromStringAndSize(NULL, len(src) << ((dstkind >> 1) - (srckind >> 1)))
    if dstkind == 2:
        interleave_bytes2(src, len(src), output)
    elif dstkind == 4:
        if srckind == 1:
            interleave_bytes4(src, len(src), output)
        else:
            interleave_bytes24(src, len(src), output)
    return output


def triage_by_release_match(
    repo: str,
    release_match: str,
    release_settings: ReleaseSettings,
    default_branches: dict[str, str],
    result: Any,
    ambiguous: dict[str, Any],
) -> Optional[Any]:
    """Check the release match of the specified `repo` and return `None` if it is not effective \
    according to `release_settings`, or decide between `result` and `ambiguous`."""
    cdef:
        Py_ssize_t str_kind = PyUnicode_KIND(<PyObject *> release_match)
        Py_ssize_t release_match_len = PyUnicode_GET_LENGTH(release_match) * str_kind
        const char *release_match_data = <const char *>PyUnicode_DATA(<PyObject *> release_match)
        PyObject *required_release_match
        const char *match_name
        int match_name_len
        const char *match_by
        int match_by_len
        long match
        long required_release_match_match
        const char *target_data
        Py_ssize_t target_len, target_kind
        PyObject *default_branch
        const char *default_branch_data
        Py_ssize_t default_branch_len, default_branch_kind, target_kind_shift
        const char *default_branch_alias_data
        const char *found
        unique_ptr[char] resolved_branch
        Py_ssize_t pos
    if (
        (
            release_match_len == rejected_name_len
            and memcmp(release_match_data, rejected_data, rejected_name_len) == 0
        )
        or
        (
            release_match_len == force_push_drop_name_len
            and memcmp(release_match_data, force_push_drop_data, force_push_drop_name_len) == 0
        )
    ):
        return result

    required_release_match = PyDict_GetItem(release_settings.native, repo)
    if required_release_match == NULL:
        # DEV-1451: if we don't have this repository in the release settings, then it is deleted
        raise AssertionError(
            f"You must take care of deleted repositories separately: {repo}",
        ) from None
    match_name = <const char *> memchr(release_match_data, ord(b"|"), release_match_len)
    if match_name == NULL:
        match_name_len = release_match_len
        match_by = release_match_data + release_match_len
        match_by_len = 0
    else:
        match_name_len = match_name - release_match_data
        match_by = match_name + str_kind
        match_by_len = release_match_len - match_name_len - str_kind
    match_name = release_match_data
    match = deref(release_match_name_to_enum)[string_view(match_name, match_name_len)]
    required_release_match_match = (<object>required_release_match).match
    if required_release_match_match != tag_or_branch:
        if match != required_release_match_match:
            return None
        dump = result
    else:
        if memcmp(match_name, b"event", 5) == 0:
            return None
        match_name_len >>= (str_kind >> 1)
        dump = ambiguous[release_match[:match_name_len]]
    if match == tag:
        target = (<object>required_release_match).tags
    elif match == branch:
        target = (<object>required_release_match).branches
    elif match == event:
        target = (<object>required_release_match).events
    else:
        raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
    target_data = <const char *> PyUnicode_DATA(<PyObject *> target)
    target_kind = PyUnicode_KIND(<PyObject *> target)
    target_len = PyUnicode_GET_LENGTH(target) << (target_kind >> 1)
    if match == branch:
        if target_kind == 1:
            default_branch_alias_data = default_branch_alias_data1
        elif target_kind == 2:
            default_branch_alias_data = default_branch_alias_data2
        else:
            default_branch_alias_data = default_branch_alias_data4
        found = <const char *>memmem(
            target_data,
            target_len,
            default_branch_alias_data,
            default_branch_alias_len << (target_kind >> 1),
        )
        if found != NULL:
            target_len -= default_branch_alias_len * target_kind
            default_branch = PyDict_GetItem(default_branches, repo)
            default_branch_len = PyUnicode_GET_LENGTH(<object> default_branch)
            default_branch_kind = PyUnicode_KIND(default_branch)
            if target_kind == default_branch_kind:
                target_kind_shift = 0
                target_len += default_branch_len << (default_branch_kind >> 1)
            elif target_kind > default_branch_kind:
                target_kind_shift = 0
                target_len += default_branch_len << ((target_kind >> 1) - (default_branch_kind >> 1))
            else:
                target_kind_shift = (default_branch_kind >> 1) - (target_kind >> 1)
                target_len <<= target_kind_shift
                target_len += default_branch_len << (default_branch_kind >> 1)
            if target_len != match_by_len:
                return None
            default_branch_data = <const char *> PyUnicode_DATA(default_branch)
            resolved_branch.reset(allocator[char]().allocate(target_len))
            pos = found - target_data
            if target_kind >= default_branch_kind:
                memcpy(resolved_branch.get(), target_data, pos)
            elif target_kind == 1:
                if default_branch_kind == 2:
                    interleave_bytes2(target_data, pos, resolved_branch.get())
                else:
                    interleave_bytes4(target_data, pos, resolved_branch.get())
            else:
                # target_kind == 2
                interleave_bytes24(target_data, pos, resolved_branch.get())
            pos <<= target_kind_shift
            if default_branch_kind >= target_kind:
                memcpy(resolved_branch.get() + pos, default_branch_data, default_branch_len << (default_branch_kind >> 1))
                pos += default_branch_len << (default_branch_kind >> 1)
            else:
                if default_branch_kind == 1:
                    if target_kind == 2:
                        interleave_bytes2(default_branch_data, default_branch_len, resolved_branch.get() + pos)
                    else:
                        interleave_bytes4(default_branch_data, default_branch_len, resolved_branch.get() + pos)
                else:
                    # default_branch_kind = 2
                    interleave_bytes24(default_branch_data, default_branch_len * 2, resolved_branch.get() + pos)
                pos += default_branch_len << (target_kind >> 1)
            if target_kind >= default_branch_kind:
                memcpy(resolved_branch.get() + pos, found + default_branch_alias_len * target_kind, target_len - pos)
            elif target_kind == 1:
                if default_branch_kind == 2:
                    interleave_bytes2(found + default_branch_alias_len, target_len - pos, resolved_branch.get() + pos)
                else:
                    interleave_bytes4(found + default_branch_alias_len, target_len - pos, resolved_branch.get() + pos)
            else:
                # target_kind == 2
                interleave_bytes24(found + default_branch_alias_len * 2, (target_len - pos) >> 1, resolved_branch.get() + pos)
            target_data = resolved_branch.get()

    if target_len != match_by_len or memcmp(target_data, match_by, match_by_len):
        return None
    return dump
