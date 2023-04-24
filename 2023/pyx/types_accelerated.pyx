# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

from cpython cimport PyDict_GetItem, PyDict_New, PyObject
from cpython.memoryview cimport PyMemoryView_Check, PyMemoryView_GET_BUFFER
from cython.operator cimport dereference, postincrement
from libc.stdint cimport int64_t

from athenian.api.native.cpython cimport (
    Py_INCREF,
    Py_None,
    Py_TYPE,
    PyByteArray_AS_STRING,
    PyByteArray_CheckExact,
    PyBytes_AS_STRING,
    PyBytes_Check,
    PyLong_AsLong,
    PyMemberDef,
    PyTypeObject,
)
from athenian.api.native.mi_heap_destroy_stl_allocator cimport (
    mi_heap_allocator_from_capsule,
    mi_heap_destroy_stl_allocator,
    mi_unordered_set,
)
from athenian.api.native.numpy cimport (
    NPY_FR_s,
    PyArray_DATA,
    PyArray_DIM,
    PyDatetimeScalarObject,
    npy_intp,
)
from athenian.api.native.optional cimport optional

from medvedi import DataFrame

from athenian.api.internal.miners.participation import PRParticipationKind
from athenian.api.models.metadata.github import (
    PullRequest,
    PullRequestComment,
    PullRequestCommit,
    PullRequestReview,
    Release,
)

pr_user_node_id_col = PullRequest.user_node_id.name
pr_merged_by_id_col = PullRequest.merged_by_id.name
release_author_node_id_col = Release.author_node_id.name
review_user_node_id_col = PullRequestReview.user_node_id.name
comment_user_node_id_col = PullRequestComment.user_node_id.name
commit_committer_user_id_col = PullRequestCommit.committer_user_id.name
commit_author_user_id_col = PullRequestCommit.author_user_id.name
values_attr = "values"

# these must be Python integers
PRParticipationKind_AUTHOR = PRParticipationKind.AUTHOR
PRParticipationKind_REVIEWER = PRParticipationKind.REVIEWER
PRParticipationKind_COMMENTER = PRParticipationKind.COMMENTER
PRParticipationKind_COMMIT_AUTHOR = PRParticipationKind.COMMIT_AUTHOR
PRParticipationKind_COMMIT_COMMITTER = PRParticipationKind.COMMIT_COMMITTER
PRParticipationKind_MERGER = PRParticipationKind.MERGER
PRParticipationKind_RELEASER = PRParticipationKind.RELEASER


cdef enum MinedPullRequestFields:
    MinedPullRequest_check_run = 0
    MinedPullRequest_comments = 1
    MinedPullRequest_commits = 2
    MinedPullRequest_deployments = 3
    MinedPullRequest_jiras = 4
    MinedPullRequest_labels = 5
    MinedPullRequest_pr = 6
    MinedPullRequest_release = 7
    MinedPullRequest_review_comments = 8
    MinedPullRequest_review_requests = 9
    MinedPullRequest_reviews = 10


cdef :
    set empty_set = set()
    Py_ssize_t df_columns_offset = (<PyTypeObject *> DataFrame).tp_members[0].offset


def extract_participant_nodes(mpr, alloc_capsule=None) -> dict:
    # the slot indexes correspond to the alphabetical order of fields
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        optional[mi_unordered_set[int64_t]] boilerplate
        PyMemberDef *mpr_slots = Py_TYPE(<PyObject *> mpr).tp_members
        PyObject *pr = dereference(
            <PyObject **> ((<char *><PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_pr].offset)
        )
        PyObject *commits = dereference(
            <PyObject **> ((<char *><PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_commits].offset)
        )
        PyObject *commits_dict = dereference(
            <PyObject **> ((<char *><PyObject *> commits) + df_columns_offset)
        )
        PyObject *commit_committers = PyDict_GetItem(
            <object>commits_dict, commit_committer_user_id_col,
        )
        PyObject *commit_authors = PyDict_GetItem(
            <object>commits_dict,  commit_author_user_id_col,
        )

        PyObject *reviews = PyDict_GetItem(
            <object> dereference(
                <PyObject **> ((<char *> dereference(
                    <PyObject **> ((<char *><PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_reviews].offset)
                )) + df_columns_offset)
            ),
            review_user_node_id_col,
        )
        PyObject *comments = PyDict_GetItem(
            <object> dereference(
                <PyObject **> ((<char *> dereference(
                        <PyObject **> ((<char *><PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_comments].offset)
                )) + df_columns_offset)
            ),
            comment_user_node_id_col,
        )
        PyObject *release = dereference(
            <PyObject **> ((<char *> <PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_release].offset)
        )
        PyObject *author = PyDict_GetItem(<object> pr, pr_user_node_id_col)
        PyObject *merger = PyDict_GetItem(<object> pr, pr_merged_by_id_col)
        PyObject *releaser = PyDict_GetItem(
            <object> dereference(
                <PyObject **> ((<char *><PyObject *> mpr) + mpr_slots[<int>MinedPullRequest_release].offset)
            ),
            release_author_node_id_col,
        )
        dict participants = PyDict_New()
        set py_boilerplate
        int64_t *data
        npy_intp i
        mi_unordered_set[int64_t].const_iterator it

    if merger == Py_None or PyLong_AsLong(merger) == 0:
        participants[PRParticipationKind_MERGER] = empty_set
    else:
        Py_INCREF(merger)
        participants[PRParticipationKind_MERGER] = {<object> merger}

    if releaser == Py_None or PyLong_AsLong(releaser) == 0:
        participants[PRParticipationKind_RELEASER] = empty_set
    else:
        Py_INCREF(releaser)
        participants[PRParticipationKind_RELEASER] = {<object> releaser}

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    boilerplate.emplace(dereference(alloc))

    data = <int64_t *> PyArray_DATA(reviews)
    for i in range(PyArray_DIM(reviews, 0)):
        dereference(boilerplate).emplace(data[i])
    dereference(boilerplate).erase(0)

    if author == Py_None or PyLong_AsLong(author) == 0:
        participants[PRParticipationKind_AUTHOR] = empty_set
    else:
        Py_INCREF(author)
        participants[PRParticipationKind_AUTHOR] = {<object> author}
        dereference(boilerplate).erase(PyLong_AsLong(author))

    py_boilerplate = set()
    it = dereference(boilerplate).const_begin()
    while it != dereference(boilerplate).const_end():
        py_boilerplate.add(dereference(it))
        postincrement(it)
    participants[PRParticipationKind_REVIEWER] = py_boilerplate

    dereference(boilerplate).clear()
    data = <int64_t *> PyArray_DATA(comments)
    for i in range(PyArray_DIM(comments, 0)):
        dereference(boilerplate).emplace(data[i])
    dereference(boilerplate).erase(0)
    py_boilerplate = set()
    it = dereference(boilerplate).const_begin()
    while it != dereference(boilerplate).const_end():
        py_boilerplate.add(dereference(it))
        postincrement(it)
    participants[PRParticipationKind_COMMENTER] = py_boilerplate

    dereference(boilerplate).clear()
    data = <int64_t *> PyArray_DATA(commit_committers)
    for i in range(PyArray_DIM(commit_committers, 0)):
        dereference(boilerplate).emplace(data[i])
    dereference(boilerplate).erase(0)
    py_boilerplate = set()
    it = dereference(boilerplate).const_begin()
    while it != dereference(boilerplate).const_end():
        py_boilerplate.add(dereference(it))
        postincrement(it)
    participants[PRParticipationKind_COMMIT_COMMITTER] = py_boilerplate

    dereference(boilerplate).clear()
    data = <int64_t *> PyArray_DATA(commit_authors)
    for i in range(PyArray_DIM(commit_authors, 0)):
        dereference(boilerplate).emplace(data[i])
    dereference(boilerplate).erase(0)
    py_boilerplate = set()
    it = dereference(boilerplate).const_begin()
    while it != dereference(boilerplate).const_end():
        py_boilerplate.add(dereference(it))
        postincrement(it)
    participants[PRParticipationKind_COMMIT_AUTHOR] = py_boilerplate

    return participants


# for field_name, (field_dtype, _) in self.dtype.fields.items():
#     if np.issubdtype(field_dtype, np.datetime64):
#         if (dt := getattr(self, field_name)) is not None and dt >= after_dt:
#             changed.append(field_name)

def find_truncated_datetime(facts, offsets, time_from) -> list:
    cdef:
        list result = []
        int64_t time_from_i64 = (<PyDatetimeScalarObject *> time_from).obval
        PyMemberDef *slots = Py_TYPE(<PyObject *> facts).tp_members
        PyObject *data_obj = dereference(
            <PyObject **> ((<char *> <PyObject *> facts) + slots[1].offset)
        )
        int64_t *offsets_i64 = <int64_t *> PyArray_DATA(<PyObject *> offsets)
        npy_intp size = PyArray_DIM(<PyObject *> offsets, 0), i
        char *data

    if PyBytes_Check(data_obj):
        data = PyBytes_AS_STRING(data_obj)
    elif PyMemoryView_Check(<object> data_obj):
        data = <char *>PyMemoryView_GET_BUFFER(<object> data_obj).buf
    elif PyByteArray_CheckExact(data_obj):
        data = PyByteArray_AS_STRING(data_obj)
    else:
        raise AssertionError(f"unsupported buffer type: {type(facts.data)}")

    assert (<PyDatetimeScalarObject *> time_from).obmeta.base == NPY_FR_s
    for i in range(size):
        if dereference(<int64_t *>(data + offsets_i64[i])) >= time_from_i64:
            result.append(i)
    return result
