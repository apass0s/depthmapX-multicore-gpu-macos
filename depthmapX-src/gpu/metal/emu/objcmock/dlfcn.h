// Test stand-in for <dlfcn.h> (Windows): dmx_metal_device_rt.cpp gets the fake
// Objective-C runtime and Metal of objc_mock.cpp instead of the system libraries.
#pragma once

#define RTLD_LAZY 1
#define RTLD_GLOBAL 8

extern "C" {
void *dlopen(const char *path, int mode);
void *dlsym(void *handle, const char *name);
char *dlerror(void);
}
