//
// Created by Henrique Borba on 19/11/2018.
//

#ifndef PHPSCI_EXT_BUFFER_H
#define PHPSCI_EXT_BUFFER_H

#include "carray.h"

/**
 * MemoryStack : The memory buffer of CArrays
 */
struct MemoryStack {
    CArray * buffer;
    int size;
    int capacity;
    size_t bsize;
} MemoryStack;


extern struct MemoryStack PHPSCI_MAIN_MEM_STACK;


void add_to_buffer(MemoryPointer * ptr, CArray array, size_t size);
void buffer_to_capacity(int new_capacity, size_t size);
void buffer_init();

#endif //PHPSCI_EXT_BUFFER_H
