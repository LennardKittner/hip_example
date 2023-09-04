#ifndef MEMSET_CMDLIST_H
#define MEMSET_CMDLIST_H
#include <cstddef>
#include "hip/hip_runtime.h"

struct Memset_cmdlist {
    size_t size;
    size_t num_groups;
    size_t num_lanes_per_group;
    size_t pattern;
    int done;
    int error;
    Memset_cmdlist *device_pointer;

    Memset_cmdlist(size_t size, size_t num_groups, size_t num_lanes_per_group, size_t pattern);
    ~Memset_cmdlist();

    hipError_t memset(size_t *to, hipStream_t stream);
};

#endif