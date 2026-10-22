// Fake Objective-C runtime and Metal for testing dmx_metal_device_rt.cpp on Windows x64
//
// dmx_metal_device_rt.cpp talks to Metal only through dlopen/dlsym and objc_msgSend.
// Here those resolve to:
//  - objc_msgSend: an assembly trampoline that looks the method up from (self, _cmd)
//    and jumps to it with the caller's arguments untouched, as the real one does; each
//    method below is a C++ function with the Objective-C method's C signature;
//  - Metal objects (device, queue, library, pipelines, buffers, command buffers,
//    encoders) on top of the CPU emulator (dmx_metal_emu.cpp, namespace dmxmetal_emu);
//    command buffers record their commands and run them on commit;
//  - Foundation's NSString / NSError and autorelease pools.
// It checks the Objective-C rules the real runtime would punish: every object has a
// reference count, a message to a released object or an unknown selector is fatal,
// autoreleasing without a pool, encoders left open, binding after endEncoding and
// committing twice are errors. At exit it prints the objects still alive per class.
//
// It does not check that the selector names and signatures are Apple's: a method is
// found here because it was written here. Those come from Apple's documentation.
//
// Environment: DMX_OBJCMOCK_FAIL = nodevice | compile | pipeline | buffer | cmdfail |
// missing:<selector> (respondsToSelector: answers NO for it); DMX_OBJCMOCK_TRACE=1
// prints every message.

#define dmxmetal dmxmetal_emu
#include "../../dmx_metal_backend.h"
#undef dmxmetal

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if !(defined(_WIN64) && (defined(__GNUC__) || defined(__clang__)))
#error "the objc mock's objc_msgSend trampoline is written for Windows x64 with GCC or Clang"
#endif

namespace {

using id = void *;
using SEL = void *;
using NSUInteger = uintptr_t;
using NSInteger = intptr_t;
using ObjcBool = signed char;
struct MTLSize {
    NSUInteger width, height, depth;
};

std::atomic<int> g_errors(0);
bool g_trace = false;
std::string g_fail;

void error(const char *format, ...) {
    va_list args;
    va_start(args, format);
    std::fprintf(stderr, "objc-mock: ERROR: ");
    std::vfprintf(stderr, format, args);
    std::fprintf(stderr, "\n");
    va_end(args);
    ++g_errors;
}

[[noreturn]] void fatal(const char *what, const char *detail) {
    std::fprintf(stderr, "objc-mock: FATAL: %s %s\n", what, detail);
    std::fflush(stderr);
    std::abort();
}

// ---------------------------------------------------------------- objects

std::mutex g_liveMutex;
std::map<std::string, int> g_live; // class -> objects alive

struct Obj {
    const char *cls;
    int refs = 1;
    bool dead = false;
    bool isClass = false;
    explicit Obj(const char *c) : cls(c) {
        std::lock_guard<std::mutex> lock(g_liveMutex);
        ++g_live[cls];
    }
    virtual ~Obj() = default;
    virtual void dealloc() {}
};

struct ClassObj : Obj {
    explicit ClassObj(const char *c) : Obj(c) { isClass = true; }
};

struct StringObj : Obj {
    std::string text;
    explicit StringObj(std::string t) : Obj("NSString"), text(std::move(t)) {}
};

struct ErrorObj : Obj {
    std::string description;
    explicit ErrorObj(std::string d) : Obj("NSError"), description(std::move(d)) {}
};

struct OptionsObj : Obj {
    bool initialised = false;
    int fastMath = -1;
    OptionsObj() : Obj("MTLCompileOptions") {}
};

struct DeviceObj;
struct LibraryObj : Obj {
    LibraryObj() : Obj("MTLLibrary") {}
};

struct FunctionObj : Obj {
    int kernel;
    explicit FunctionObj(int k) : Obj("MTLFunction"), kernel(k) {}
};

struct PipelineObj : Obj {
    int kernel;
    uint32_t maxThreads;
    PipelineObj(int k, uint32_t t) : Obj("MTLComputePipelineState"), kernel(k), maxThreads(t) {}
};

struct QueueObj : Obj {
    DeviceObj *device;
    explicit QueueObj(DeviceObj *d) : Obj("MTLCommandQueue"), device(d) {}
};

struct BufferObj : Obj {
    DeviceObj *device;
    dmxmetal_emu::Buffer *buffer;
    BufferObj(DeviceObj *d, dmxmetal_emu::Buffer *b) : Obj("MTLBuffer"), device(d), buffer(b) {}
    void dealloc() override;
};

struct EncoderObj;
struct CommandBufferObj : Obj {
    QueueObj *queue;
    EncoderObj *encoder = nullptr; // the open encoder (not retained, as in Metal)
    std::vector<std::function<void(dmxmetal_emu::Device &)>> commands;
    std::vector<id> retained; // buffers and pipelines bound, held until completion
    bool committed = false;
    NSUInteger status = 0; // MTLCommandBufferStatusNotEnqueued
    ErrorObj *error = nullptr;
    explicit CommandBufferObj(QueueObj *q) : Obj("MTLCommandBuffer"), queue(q) {}
    void dealloc() override;
};

struct EncoderObj : Obj {
    CommandBufferObj *commands;
    bool ended = false;
    int pipeline = -1;
    explicit EncoderObj(CommandBufferObj *c) : Obj("MTLComputeCommandEncoder"), commands(c) {}
    void dealloc() override {
        if (!ended) {
            error("compute encoder deallocated without endEncoding");
        }
    }
};

struct DeviceObj : Obj {
    std::unique_ptr<dmxmetal_emu::Device> emu;
    explicit DeviceObj(std::unique_ptr<dmxmetal_emu::Device> e) : Obj("MTLDevice"), emu(std::move(e)) {}
};

void BufferObj::dealloc() { device->emu->release(buffer); }

ClassObj g_NSString("+NSString");
ClassObj g_MTLCompileOptions("+MTLCompileOptions");

Obj *obj(id self, const char *selector) {
    Obj *o = static_cast<Obj *>(self);
    if (!o) {
        fatal("message to nil from this code path (objc would ignore it, but rt never sends one):", selector);
    }
    if (o->dead) {
        fatal("message to a deallocated object:", selector);
    }
    return o;
}

void destroy(Obj *o) {
    o->dealloc();
    o->dead = true; // kept as a zombie, so later messages are caught
    std::lock_guard<std::mutex> lock(g_liveMutex);
    --g_live[o->cls];
}

void releaseObj(Obj *o) {
    if (o->isClass) {
        return;
    }
    if (--o->refs == 0) {
        destroy(o);
    } else if (o->refs < 0) {
        fatal("over-release of", o->cls);
    }
}

// ---------------------------------------------------------------- autorelease pools

thread_local std::vector<Obj *> t_autoreleased;
thread_local std::vector<size_t> t_pools;

id autorelease(Obj *o) {
    if (t_pools.empty()) {
        error("%s autoreleased with no pool in place (leaks)", o->cls);
    }
    t_autoreleased.push_back(o);
    return o;
}

void *poolPush() {
    t_pools.push_back(t_autoreleased.size());
    return reinterpret_cast<void *>(uintptr_t(t_pools.size()));
}

void poolPop(void *token) {
    const size_t depth = size_t(reinterpret_cast<uintptr_t>(token));
    if (depth == 0 || depth != t_pools.size()) {
        error("autorelease pools popped out of order");
        if (depth == 0 || depth > t_pools.size()) {
            return;
        }
    }
    const size_t mark = t_pools[depth - 1];
    t_pools.resize(depth - 1);
    while (t_autoreleased.size() > mark) {
        Obj *o = t_autoreleased.back();
        t_autoreleased.pop_back();
        releaseObj(o);
    }
}

// ---------------------------------------------------------------- methods

std::map<std::string, void *> g_methods; // "<class> <selector>" -> function
std::set<std::string> g_selectors;       // interned selector names; SEL = c_str()
std::mutex g_selectorMutex;

const char *name(SEL sel) { return static_cast<const char *>(sel); }

void *findMethod(Obj *o, const char *selector) {
    auto it = g_methods.find(std::string(o->cls) + " " + selector);
    if (it == g_methods.end() && !o->isClass) {
        it = g_methods.find(std::string("* ") + selector);
    }
    return it == g_methods.end() ? nullptr : it->second;
}

// NSObject
id m_retain(id self, SEL) {
    ++obj(self, "retain")->refs;
    return self;
}
void m_release(id self, SEL) { releaseObj(obj(self, "release")); }
ObjcBool m_respondsToSelector(id self, SEL, SEL query) {
    if (g_fail == std::string("missing:") + name(query)) {
        return 0;
    }
    return findMethod(obj(self, "respondsToSelector:"), name(query)) ? 1 : 0;
}

// Foundation
id m_stringWithUTF8String(id, SEL, const char *text) {
    return text ? autorelease(new StringObj(text)) : nullptr;
}
const char *m_UTF8String(id self, SEL) { return static_cast<StringObj *>(obj(self, "UTF8String"))->text.c_str(); }
id m_localizedDescription(id self, SEL) {
    return autorelease(new StringObj(static_cast<ErrorObj *>(obj(self, "localizedDescription"))->description));
}
void setError(id *out, const std::string &description) {
    if (out) {
        *out = autorelease(new ErrorObj(description));
    }
}

// MTLCompileOptions
id m_alloc(id, SEL) { return new OptionsObj(); }
id m_init(id self, SEL) {
    static_cast<OptionsObj *>(obj(self, "init"))->initialised = true;
    return self;
}
void m_setFastMathEnabled(id self, SEL, ObjcBool value) {
    static_cast<OptionsObj *>(obj(self, "setFastMathEnabled:"))->fastMath = value ? 1 : 0;
}

// MTLDevice
DeviceObj *device(id self, const char *selector) { return static_cast<DeviceObj *>(obj(self, selector)); }

id m_name(id self, SEL) {
    device(self, "name");
    return autorelease(new StringObj("Metal on the objc mock (CPU emulator)"));
}
ObjcBool m_supportsFamily(id self, SEL, NSInteger family) {
    device(self, "supportsFamily:");
    return ((family >= 1001 && family <= 1007) || family == 2002) ? 1 : 0; // like an Apple7 GPU (M1)
}
uint64_t m_recommendedMaxWorkingSetSize(id self, SEL) {
    return device(self, "recommendedMaxWorkingSetSize")->emu->workingSetBytes();
}
NSUInteger m_currentAllocatedSize(id self, SEL) { return device(self, "currentAllocatedSize")->emu->allocatedBytes(); }
NSUInteger m_maxBufferLength(id self, SEL) { return device(self, "maxBufferLength")->emu->maxBufferBytes(); }
NSUInteger m_maxThreadgroupMemoryLength(id self, SEL) {
    return device(self, "maxThreadgroupMemoryLength")->emu->maxThreadgroupMemory();
}
id m_newCommandQueue(id self, SEL) { return new QueueObj(device(self, "newCommandQueue")); }

const char *const kKernelNames[dmxmetal_emu::kKernelCount] = {"dmx_probe",
                                                              "dmx_vis_local_rows",
                                                              "dmx_vis_global_pull",
                                                              "dmx_vis_global_downsweep",
                                                              "dmx_vis_global_combine",
                                                              "dmx_vis_global_count"};

id m_newLibraryWithSource(id self, SEL, id source, id options, id *err) {
    device(self, "newLibraryWithSource:options:error:");
    Obj *s = obj(source, "newLibraryWithSource:options:error: (source)");
    Obj *o = obj(options, "newLibraryWithSource:options:error: (options)");
    if (std::strcmp(s->cls, "NSString") != 0 || std::strcmp(o->cls, "MTLCompileOptions") != 0) {
        error("newLibraryWithSource: wrong argument classes %s, %s", s->cls, o->cls);
    }
    auto *opts = static_cast<OptionsObj *>(o);
    if (!opts->initialised || opts->fastMath != 0) {
        error("MTLCompileOptions not initialised or fast math not disabled");
    }
    const std::string &text = static_cast<StringObj *>(s)->text;
    for (const char *kernel : kKernelNames) {
        if (text.find(std::string("kernel void ") + kernel) == std::string::npos &&
            text.find(kernel) == std::string::npos) {
            error("kernel source lacks %s", kernel);
        }
    }
    if (g_fail == "compile") {
        setError(err, "program_source:1:1: error: simulated compiler failure");
        return nullptr;
    }
    return new LibraryObj();
}
id m_newFunctionWithName(id self, SEL, id functionName) {
    obj(self, "newFunctionWithName:");
    const std::string &n = static_cast<StringObj *>(obj(functionName, "newFunctionWithName: (name)"))->text;
    for (int k = 0; k < dmxmetal_emu::kKernelCount; ++k) {
        if (n == kKernelNames[k]) {
            return new FunctionObj(k);
        }
    }
    return nullptr;
}
id m_newComputePipelineStateWithFunction(id self, SEL, id function, id *err) {
    DeviceObj *d = device(self, "newComputePipelineStateWithFunction:error:");
    auto *f = static_cast<FunctionObj *>(obj(function, "newComputePipelineStateWithFunction:error: (function)"));
    if (g_fail == "pipeline") {
        setError(err, "simulated pipeline failure");
        return nullptr;
    }
    return new PipelineObj(f->kernel, d->emu->maxThreadsPerThreadgroup(dmxmetal_emu::Kernel(f->kernel)));
}
id m_newBufferWithLength(id self, SEL, NSUInteger length, NSUInteger options) {
    DeviceObj *d = device(self, "newBufferWithLength:options:");
    if (options != 0) {
        error("buffer options %llu, expected MTLResourceStorageModeShared (0)", (unsigned long long)options);
    }
    if (g_fail == "buffer" && length > 16) {
        return nullptr;
    }
    std::string why;
    dmxmetal_emu::Buffer *b = d->emu->allocate(length, why);
    return b ? new BufferObj(d, b) : nullptr;
}

// MTLComputePipelineState, MTLBuffer
NSUInteger m_maxTotalThreadsPerThreadgroup(id self, SEL) {
    return static_cast<PipelineObj *>(obj(self, "maxTotalThreadsPerThreadgroup"))->maxThreads;
}
void *m_contents(id self, SEL) {
    auto *b = static_cast<BufferObj *>(obj(self, "contents"));
    return b->device->emu->contents(b->buffer);
}

// MTLCommandQueue, MTLCommandBuffer
id m_commandBuffer(id self, SEL) {
    return autorelease(new CommandBufferObj(static_cast<QueueObj *>(obj(self, "commandBuffer"))));
}
CommandBufferObj *commands(id self, const char *selector) { return static_cast<CommandBufferObj *>(obj(self, selector)); }

id m_computeCommandEncoder(id self, SEL) {
    CommandBufferObj *c = commands(self, "computeCommandEncoder");
    if (c->committed || (c->encoder && !c->encoder->ended)) {
        error("computeCommandEncoder on a committed command buffer or with an encoder open");
    }
    c->encoder = new EncoderObj(c);
    return autorelease(c->encoder);
}
void m_commit(id self, SEL) {
    CommandBufferObj *c = commands(self, "commit");
    if (c->committed) {
        error("command buffer committed twice");
        return;
    }
    if (c->encoder && !c->encoder->ended) {
        error("command buffer committed with an encoder open");
    }
    c->committed = true;
    // Runs now: a valid schedule, as command buffers run in commit order.
    dmxmetal_emu::Device &emu = *c->queue->device->emu;
    std::string why;
    bool ok = emu.begin(why);
    if (ok) {
        for (auto &command : c->commands) {
            command(emu);
        }
        ok = emu.commit(true, why);
    }
    if (ok && g_fail == "cmdfail") {
        ok = false;
        why = "simulated GPU fault";
    }
    c->status = ok ? 4 : 5; // MTLCommandBufferStatusCompleted / Error
    if (!ok) {
        c->error = new ErrorObj("Execution of the command buffer was aborted (" + why + ")");
    }
    for (id r : c->retained) {
        releaseObj(static_cast<Obj *>(r));
    }
    c->retained.clear();
    c->commands.clear();
}
void CommandBufferObj::dealloc() {
    for (id r : retained) {
        releaseObj(static_cast<Obj *>(r));
    }
    if (error) {
        releaseObj(error);
    }
}
void m_waitUntilCompleted(id self, SEL) {
    if (!commands(self, "waitUntilCompleted")->committed) {
        error("waitUntilCompleted on a command buffer never committed (would block forever)");
    }
}
NSUInteger m_status(id self, SEL) { return commands(self, "status")->status; }
id m_error(id self, SEL) { return commands(self, "error")->error; }

// MTLComputeCommandEncoder
EncoderObj *encoder(id self, const char *selector) {
    auto *e = static_cast<EncoderObj *>(obj(self, selector));
    if (e->ended) {
        error("%s after endEncoding", selector);
    }
    return e;
}
void hold(CommandBufferObj *c, id object) {
    ++static_cast<Obj *>(object)->refs;
    c->retained.push_back(object);
}
void m_setBuffer(id self, SEL, id buffer, NSUInteger offset, NSUInteger index) {
    EncoderObj *e = encoder(self, "setBuffer:offset:atIndex:");
    auto *b = static_cast<BufferObj *>(obj(buffer, "setBuffer:offset:atIndex: (buffer)"));
    hold(e->commands, b);
    e->commands->commands.push_back(
        [b, offset, index](dmxmetal_emu::Device &emu) { emu.setBuffer(b->buffer, size_t(offset), int(index)); });
}
void m_setBytes(id self, SEL, const void *bytes, NSUInteger length, NSUInteger index) {
    EncoderObj *e = encoder(self, "setBytes:length:atIndex:");
    auto copy = std::make_shared<std::vector<unsigned char>>(static_cast<const unsigned char *>(bytes),
                                                             static_cast<const unsigned char *>(bytes) + length);
    e->commands->commands.push_back(
        [copy, index](dmxmetal_emu::Device &emu) { emu.setBytes(copy->data(), copy->size(), int(index)); });
}
void m_setThreadgroupMemoryLength(id self, SEL, NSUInteger length, NSUInteger index) {
    EncoderObj *e = encoder(self, "setThreadgroupMemoryLength:atIndex:");
    e->commands->commands.push_back(
        [length, index](dmxmetal_emu::Device &emu) { emu.setThreadgroupMemory(size_t(length), int(index)); });
}
void m_setComputePipelineState(id self, SEL, id pipeline) {
    EncoderObj *e = encoder(self, "setComputePipelineState:");
    auto *p = static_cast<PipelineObj *>(obj(pipeline, "setComputePipelineState: (pipeline)"));
    hold(e->commands, p);
    e->pipeline = p->kernel;
}
void m_dispatchThreadgroups(id self, SEL, MTLSize groups, MTLSize threads) {
    EncoderObj *e = encoder(self, "dispatchThreadgroups:threadsPerThreadgroup:");
    if (e->pipeline < 0) {
        error("dispatch without a pipeline state");
        return;
    }
    if (groups.depth != 1 || threads.height != 1 || threads.depth != 1 || groups.width > 0xFFFFFFFFu ||
        groups.height > 0xFFFFFFFFu || threads.width > 0xFFFFFFFFu) {
        error("unexpected dispatch shape");
    }
    const auto kernel = dmxmetal_emu::Kernel(e->pipeline);
    const uint32_t gx = uint32_t(groups.width), gy = uint32_t(groups.height), t = uint32_t(threads.width);
    e->commands->commands.push_back([kernel, gx, gy, t](dmxmetal_emu::Device &emu) { emu.dispatch(kernel, gx, gy, t); });
}
void m_endEncoding(id self, SEL) { encoder(self, "endEncoding")->ended = true; }

// Metal's C entry point
id createSystemDefaultDevice() {
    if (g_fail == "nodevice") {
        return nullptr;
    }
    int code = 0;
    std::string why;
    std::unique_ptr<dmxmetal_emu::Device> emu = dmxmetal_emu::createDevice(&code, why);
    return emu ? new DeviceObj(std::move(emu)) : nullptr;
}

id getClass(const char *className) {
    if (std::strcmp(className, "NSString") == 0) {
        return &g_NSString;
    }
    if (std::strcmp(className, "MTLCompileOptions") == 0) {
        return &g_MTLCompileOptions;
    }
    return nullptr;
}

SEL registerName(const char *selector) {
    std::lock_guard<std::mutex> lock(g_selectorMutex);
    return const_cast<char *>(g_selectors.insert(selector).first->c_str());
}

template <typename F> void method(const char *cls, const char *selector, F f) {
    g_methods[std::string(cls) + " " + selector] = reinterpret_cast<void *>(f);
}

struct Setup {
    Setup() {
        const char *fail = std::getenv("DMX_OBJCMOCK_FAIL");
        g_fail = fail ? fail : "";
        const char *trace = std::getenv("DMX_OBJCMOCK_TRACE");
        g_trace = trace && *trace == '1';

        method("*", "retain", m_retain);
        method("*", "release", m_release);
        method("*", "respondsToSelector:", m_respondsToSelector);
        method("+NSString", "stringWithUTF8String:", m_stringWithUTF8String);
        method("+NSString", "respondsToSelector:", m_respondsToSelector);
        method("NSString", "UTF8String", m_UTF8String);
        method("NSError", "localizedDescription", m_localizedDescription);
        method("+MTLCompileOptions", "alloc", m_alloc);
        method("MTLCompileOptions", "init", m_init);
        method("MTLCompileOptions", "setFastMathEnabled:", m_setFastMathEnabled);
        method("MTLDevice", "name", m_name);
        method("MTLDevice", "supportsFamily:", m_supportsFamily);
        method("MTLDevice", "recommendedMaxWorkingSetSize", m_recommendedMaxWorkingSetSize);
        method("MTLDevice", "currentAllocatedSize", m_currentAllocatedSize);
        method("MTLDevice", "maxBufferLength", m_maxBufferLength);
        method("MTLDevice", "maxThreadgroupMemoryLength", m_maxThreadgroupMemoryLength);
        method("MTLDevice", "newCommandQueue", m_newCommandQueue);
        method("MTLDevice", "newLibraryWithSource:options:error:", m_newLibraryWithSource);
        method("MTLDevice", "newComputePipelineStateWithFunction:error:", m_newComputePipelineStateWithFunction);
        method("MTLDevice", "newBufferWithLength:options:", m_newBufferWithLength);
        method("MTLLibrary", "newFunctionWithName:", m_newFunctionWithName);
        method("MTLComputePipelineState", "maxTotalThreadsPerThreadgroup", m_maxTotalThreadsPerThreadgroup);
        method("MTLBuffer", "contents", m_contents);
        method("MTLCommandQueue", "commandBuffer", m_commandBuffer);
        method("MTLCommandBuffer", "computeCommandEncoder", m_computeCommandEncoder);
        method("MTLCommandBuffer", "commit", m_commit);
        method("MTLCommandBuffer", "waitUntilCompleted", m_waitUntilCompleted);
        method("MTLCommandBuffer", "status", m_status);
        method("MTLCommandBuffer", "error", m_error);
        method("MTLComputeCommandEncoder", "setBuffer:offset:atIndex:", m_setBuffer);
        method("MTLComputeCommandEncoder", "setBytes:length:atIndex:", m_setBytes);
        method("MTLComputeCommandEncoder", "setThreadgroupMemoryLength:atIndex:", m_setThreadgroupMemoryLength);
        method("MTLComputeCommandEncoder", "setComputePipelineState:", m_setComputePipelineState);
        method("MTLComputeCommandEncoder", "dispatchThreadgroups:threadsPerThreadgroup:", m_dispatchThreadgroups);
        method("MTLComputeCommandEncoder", "endEncoding", m_endEncoding);
    }
    ~Setup() { // static destruction: report what is still alive
        std::string live;
        for (const auto &c : g_live) {
            if (c.second != 0 && c.first[0] != '+') {
                live += " " + c.first + "=" + std::to_string(c.second);
            }
        }
        std::fprintf(stderr, "objc-mock: exit: %d error(s); objects alive:%s\n", g_errors.load(),
                     live.empty() ? " none" : live.c_str());
    }
};

Setup g_setup;

} // namespace

// ---------------------------------------------------------------- C entry points

extern "C" void *dmx_objcmock_lookup(void *self, void *sel) {
    Obj *o = obj(self, name(sel));
    void *f = findMethod(o, name(sel));
    if (g_trace) {
        std::fprintf(stderr, "objc-mock: [%s %s]\n", o->cls, name(sel));
    }
    if (!f) {
        fatal(o->cls, (std::string("does not recognize selector ") + name(sel)).c_str());
    }
    return f;
}

// objc_msgSend: keeps the four argument registers (and the stack arguments, untouched),
// looks the method up, restores them and jumps: the method returns straight to the caller.
extern "C" void dmx_objcmock_msgSend();
asm(R"(
    .text
    .globl dmx_objcmock_msgSend
    .p2align 4
dmx_objcmock_msgSend:
    pushq %rcx
    pushq %rdx
    pushq %r8
    pushq %r9
    subq $40, %rsp
    callq dmx_objcmock_lookup
    addq $40, %rsp
    popq %r9
    popq %r8
    popq %rdx
    popq %rcx
    jmpq *%rax
)");

namespace {
int g_handleObjc, g_handleFoundation, g_handleMetal;
}

extern "C" void *dlopen(const char *path, int) {
    if (std::strcmp(path, "/usr/lib/libobjc.A.dylib") == 0) {
        return &g_handleObjc;
    }
    if (std::strcmp(path, "/System/Library/Frameworks/Foundation.framework/Foundation") == 0) {
        return &g_handleFoundation;
    }
    if (std::strcmp(path, "/System/Library/Frameworks/Metal.framework/Metal") == 0) {
        return &g_handleMetal;
    }
    error("dlopen of an unexpected path %s", path);
    return nullptr;
}

extern "C" void *dlsym(void *handle, const char *symbol) {
    if (handle == &g_handleObjc) {
        if (!std::strcmp(symbol, "objc_getClass")) return reinterpret_cast<void *>(getClass);
        if (!std::strcmp(symbol, "sel_registerName")) return reinterpret_cast<void *>(registerName);
        if (!std::strcmp(symbol, "objc_msgSend")) return reinterpret_cast<void *>(dmx_objcmock_msgSend);
        if (!std::strcmp(symbol, "objc_autoreleasePoolPush")) return reinterpret_cast<void *>(poolPush);
        if (!std::strcmp(symbol, "objc_autoreleasePoolPop")) return reinterpret_cast<void *>(poolPop);
    }
    if (handle == &g_handleMetal && !std::strcmp(symbol, "MTLCreateSystemDefaultDevice")) {
        return reinterpret_cast<void *>(createSystemDefaultDevice);
    }
    error("dlsym of an unexpected symbol %s", symbol);
    return nullptr;
}

extern "C" char *dlerror(void) { return nullptr; }
