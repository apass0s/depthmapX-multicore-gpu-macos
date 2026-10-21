// depthmapX GPU module - run-time loader for depthmapX_gpu.dll

#include "gpu/host/gpuloader.h"

#include <cstdio>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#include <climits>
#include <cstdlib>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

namespace dmxaccel {

namespace {

#ifdef _WIN32
const char *const kLibraryName = "depthmapX_gpu.dll";

std::string executableDirectory() {
    wchar_t path[MAX_PATH * 4];
    DWORD len = GetModuleFileNameW(nullptr, path, DWORD(sizeof(path) / sizeof(path[0])));
    if (len == 0 || len >= sizeof(path) / sizeof(path[0])) {
        return std::string();
    }
    std::wstring dir(path, len);
    dir = dir.substr(0, dir.find_last_of(L"\\/") + 1);
    int bytes = WideCharToMultiByte(CP_UTF8, 0, dir.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string out(size_t(bytes > 0 ? bytes - 1 : 0), '\0');
    WideCharToMultiByte(CP_UTF8, 0, dir.c_str(), -1, &out[0], bytes, nullptr, nullptr);
    return out;
}

void *openLibrary(const std::string &utf8Path, std::string &error) {
    int chars = MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, nullptr, 0);
    std::wstring wide(size_t(chars > 0 ? chars - 1 : 0), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.c_str(), -1, &wide[0], chars);
    // Full path + altered search path: dependencies resolve next to the DLL, not from the CWD.
    HMODULE h = LoadLibraryExW(wide.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!h) {
        error = "cannot load " + utf8Path + " (Windows error " + std::to_string(GetLastError()) + ")";
    }
    return reinterpret_cast<void *>(h);
}

void *findSymbol(void *handle, const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
}
#elif defined(__APPLE__)
const char *const kLibraryName = "libdepthmapX_gpu.dylib"; // the Metal module (gpu/metal)

std::string executableDirectory() {
    char raw[PATH_MAX * 2];
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) != 0) {
        return std::string();
    }
    char resolved[PATH_MAX];
    if (!realpath(raw, resolved)) {
        return std::string();
    }
    std::string dir(resolved);
    return dir.substr(0, dir.find_last_of('/') + 1);
}
#else
const char *const kLibraryName = "libdepthmapX_gpu.so";

std::string executableDirectory() {
    char path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) {
        return std::string();
    }
    std::string dir(path, size_t(len));
    return dir.substr(0, dir.find_last_of('/') + 1);
}
#endif

#ifndef _WIN32
void *openLibrary(const std::string &path, std::string &error) {
    void *h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char *msg = dlerror();
        error = "cannot load " + path + (msg ? std::string(": ") + msg : std::string());
    }
    return h;
}

void *findSymbol(void *handle, const char *name) { return dlsym(handle, name); }
#endif

template <typename Fn> bool resolve(void *handle, const char *name, Fn &fn, std::string &missing) {
    fn = reinterpret_cast<Fn>(findSymbol(handle, name));
    if (!fn) {
        missing = name;
    }
    return fn != nullptr;
}

} // namespace

GpuModule &GpuModule::instance() {
    static GpuModule module; // thread-safe initialisation
    return module;
}

GpuModule::GpuModule() { m_available = load(); }

bool GpuModule::load() {
    std::string dir = executableDirectory();
    if (dir.empty()) {
        m_reason = "cannot determine the executable directory";
        return false;
    }
    m_handle = openLibrary(dir + kLibraryName, m_reason);
#ifdef __APPLE__
    if (!m_handle) {
        // depthmapX.app/Contents/MacOS/depthmapX -> depthmapX.app/Contents/Frameworks/
        std::string firstError = m_reason;
        m_handle = openLibrary(dir + "../Frameworks/" + kLibraryName, m_reason);
        if (!m_handle) {
            m_reason = firstError;
        }
    }
#endif
    if (!m_handle) {
        return false;
    }

    std::string missing;
    bool ok = resolve(m_handle, "dmx_gpu_api_version", m_apiVersion, missing) &&
              resolve(m_handle, "dmx_gpu_last_error", m_lastError, missing) &&
              resolve(m_handle, "dmx_gpu_probe", m_probe, missing);
    if (!ok) {
        m_reason = std::string(kLibraryName) + " has no symbol " + missing;
        return false;
    }
    int version = m_apiVersion();
    if (version != DMX_GPU_API_VERSION) {
        m_reason = std::string(kLibraryName) + " has API version " + std::to_string(version) + ", expected " +
                   std::to_string(DMX_GPU_API_VERSION);
        return false;
    }
    ok = resolve(m_handle, "dmx_vis_local_create", visLocalCreate, missing) &&
         resolve(m_handle, "dmx_vis_local_run", visLocalRun, missing) &&
         resolve(m_handle, "dmx_vis_local_destroy", visLocalDestroy, missing) &&
         resolve(m_handle, "dmx_vis_global_create", visGlobalCreate, missing) &&
         resolve(m_handle, "dmx_vis_global_lanes", visGlobalLanes, missing) &&
         resolve(m_handle, "dmx_vis_global_run", visGlobalRun, missing) &&
         resolve(m_handle, "dmx_vis_global_destroy", visGlobalDestroy, missing);
    if (!ok) {
        m_reason = std::string(kLibraryName) + " has no symbol " + missing;
        return false;
    }

    if (m_probe(&m_device) != DMX_GPU_OK) {
        m_reason = m_lastError();
        return false;
    }
    return true;
}

std::string GpuModule::deviceDescription() const {
    char buf[400];
#ifdef __APPLE__
    // Metal: cc_major is the newest Apple GPU family supported, total_mem the recommended
    // working set of the unified memory
    std::snprintf(buf, sizeof(buf), "%s, Metal, GPU family Apple%d, %.0f MiB", m_device.name, m_device.cc_major,
                  m_device.total_mem / 1048576.0);
#else
    std::snprintf(buf, sizeof(buf), "%s, compute %d.%d, %.0f MiB", m_device.name, m_device.cc_major,
                  m_device.cc_minor, m_device.total_mem / 1048576.0);
#endif
    return buf;
}

std::string GpuModule::lastError() const { return m_lastError ? std::string(m_lastError()) : std::string(); }

} // namespace dmxaccel
