# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: libraries = sentry
# distutils: runtime_library_dirs = /usr/local/lib


cdef extern from "sentry.h" nogil:
    struct sentry_options_s
    ctypedef sentry_options_s sentry_options_t
    sentry_options_t *sentry_options_new()
    void sentry_options_set_dsn(sentry_options_t *opts, const char *dsn)
    void sentry_options_set_release(sentry_options_t *opts, const char *release)
    void sentry_options_set_environment(sentry_options_t *opts, const char *environment)
    void sentry_options_set_debug(sentry_options_t *opts, int debug)
    void sentry_options_set_max_breadcrumbs(sentry_options_t *opts, size_t max_breadcrumbs)
    void sentry_options_set_handler_path(sentry_options_t *opts, const char *path)
    void sentry_options_set_symbolize_stacktraces(sentry_options_t *opts, int val)
    int sentry_init(sentry_options_t *options)
    int sentry_close()


def init(str dsn not None, str release not None, str env not None) -> None:
    cdef:
        sentry_options_t *options = sentry_options_new()
    sentry_options_set_dsn(options, dsn.encode())
    sentry_options_set_release(options, release.encode())
    sentry_options_set_environment(options, env.encode())
    sentry_options_set_debug(options, env != "production")
    sentry_options_set_handler_path(options, b"/usr/local/bin/crashpad_handler")
    sentry_options_set_symbolize_stacktraces(options, 1)
    sentry_options_set_max_breadcrumbs(options, 20)
    sentry_init(options)


def fini():
    sentry_close()
