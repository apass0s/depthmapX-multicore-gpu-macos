// The CPU emulator's device layer in its own namespace, so it can sit under the fake
// Metal of objc_mock.cpp in the same library as dmx_metal_device_rt.cpp (which defines
// dmxmetal::Buffer and dmxmetal::createDevice itself).
#define dmxmetal dmxmetal_emu
#include "../dmx_metal_emu.cpp"
