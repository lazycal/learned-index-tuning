#pragma once

#include <cstddef>
#include <cstdint>
namespace rmi_universal {
bool load(char const* dataPath);
void cleanup();
// size_t RMI_SIZE;
// const uint64_t BUILD_TIME_NS = 0;
const char NAME[] = "rmi_universal";
uint64_t lookup(uint64_t key, size_t* err);
}
