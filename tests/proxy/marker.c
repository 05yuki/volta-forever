// The functions by which a program can tell that the layer is there (crates/vf-proxy/src/marker.rs):
//   marker PROXY_PATH VERSION
// opens the layer's libcuda.so.1, looks them up as a program would (dlsym), and compares what
// they give with the interface number 1 and with VERSION. Exit status 1 when something differs.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: marker PROXY_PATH VERSION\n");
        return 2;
    }
    void *layer = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!layer) { fprintf(stderr, "cannot open %s: %s\n", argv[1], dlerror()); return 2; }
    int wrong = 0;
    int (*abi)(void) = (int (*)(void))dlsym(layer, "volta_forever_abi");
    const char *(*version)(void) = (const char *(*)(void))dlsym(layer, "volta_forever_version");
    if (!abi || abi() != 1) {
        wrong++;
        printf("WRONG  volta_forever_abi: %s\n", abi ? "not 1" : "not found");
    } else printf("ok     volta_forever_abi() = %d\n", abi());
    if (!version || strcmp(version(), argv[2])) {
        wrong++;
        printf("WRONG  volta_forever_version: %s (want %s)\n", version ? version() : "not found", argv[2]);
    } else printf("ok     volta_forever_version() = %s\n", version());
    // Found among the symbols of the process too, once the library is loaded for all.
    if (!abi || dlsym(RTLD_DEFAULT, "volta_forever_abi") != (void *)abi) {
        wrong++;
        printf("WRONG  volta_forever_abi is not found among the symbols of the process\n");
    } else printf("ok     found among the symbols of the process\n");
    return wrong ? 1 : 0;
}
