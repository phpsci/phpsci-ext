#ifndef PHPSCI_EXT_BUFFER_H
#define PHPSCI_EXT_BUFFER_H

#include "carray.h"

/**
 * MemoryStack : The memory buffer of CArrays
 */
struct MemoryStack {
    CArray ** buffer;
    int size;
    int capacity;
    int freed;
    size_t bsize;
} MemoryStack;


extern struct MemoryStack PHPSCI_MAIN_MEM_STACK;


void add_to_buffer(MemoryPointer *out, CArray * array, size_t size);
void buffer_to_capacity(int new_capacity, size_t size);
void remove_from_buffer(CArray * ptr);
void buffer_init();
void buffer_remove(CArray * ptr);
CArray* get_from_buffer(int index);
void buffer_remove_index(int index);
#endif //PHPSCI_EXT_BUFFER_H