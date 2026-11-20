// Runtime of a forwarding library (the generated part follows). Nothing but the C library is used.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HIDDEN __attribute__((visibility("hidden")))
#define EXPORT __attribute__((visibility("default")))

HIDDEN void *vf_slots[VF_NSLOTS + 1];
extern const char *const vf_targets[];
extern const char *const vf_notes[];
#ifdef VF_HAS_STATUS
extern const int vf_status_ns;
#endif
#ifndef VF_HAS_STATUS
__attribute__((unused)) static const int vf_status_ns = -1;
#endif

static void *handle;
static pthread_mutex_t lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;

// The 12.9 library, loaded by the first call of any function and never again: dlopen hands back the
// same object for the same name, and a recursive lock covers a constructor of that library that calls
// back into this one. By name, so the loader's search path and the audit library's map apply.
static void *library(void) {
    void *h = __atomic_load_n(&handle, __ATOMIC_ACQUIRE);
    if (h) return h;
    pthread_mutex_lock(&lock);
    h = handle;
    if (!h) {
        const char *path = getenv(VF_ENV);
        if (!path || !*path) path = VF_TARGET;
        h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "[volta-forever] %s: cannot load %s: %s\n", VF_LIB, path, dlerror());
            abort();
        }
        if (getenv("VF_FWD_VERBOSE")) fprintf(stderr, "[volta-forever] %s: forwarding to %s\n", VF_LIB, path);
        __atomic_store_n(&handle, h, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&lock);
    return h;
}

// First call of the function in slot `i`. Two threads may both get here: both store the same address.
HIDDEN void *vf_fwd_resolve(unsigned long i) {
    const char *name = vf_targets[i];
    void *p = dlsym(library(), name);
    if (!p) {
        fprintf(stderr, "[volta-forever] %s: %s is missing from the library it forwards to\n", VF_LIB, name);
        abort();
    }
    if (vf_notes[i]) fprintf(stderr, "[volta-forever] %s: %s\n", VF_LIB, vf_notes[i]);
    if (getenv("VF_FWD_VERBOSE")) fprintf(stderr, "[volta-forever] %s: bound %s\n", VF_LIB, name);
    __atomic_store_n(&vf_slots[i], p, __ATOMIC_RELEASE);
    return p;
}

// A function the 12.9 library cannot serve is said once, on the first call.
static void vf_unsupported(char *said, const char *name, const char *why) {
    if (!__atomic_exchange_n(said, 1, __ATOMIC_RELAXED))
        fprintf(stderr, "[volta-forever] %s: %s is not supported: %s\n", VF_LIB, name, why);
}
