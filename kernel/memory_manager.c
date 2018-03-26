/*
  +----------------------------------------------------------------------+
  | PHP Version 7 | PHPSci                                               |
  +----------------------------------------------------------------------+
  | Copyright (c) 2018 Henrique Borba                                    |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: Henrique Borba <henrique.borba.dev@gmail.com>                |
  +----------------------------------------------------------------------+
*/

#include "memory_manager.h"
#include "carray.h"
#include "../phpsci.h"

/**
 * MEMORY STACK
 *
 * CArrays Memory Buffer
 */
struct MemoryStack PHPSCI_MAIN_MEM_STACK = {UNINITIALIZED,UNINITIALIZED,UNINITIALIZED, UNINITIALIZED};

/**
 * Initialize MemoryStack Buffer
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @todo Same from buffer_to_capacity
 */
void stack_init(size_t size) {
    PHPSCI_MAIN_MEM_STACK.size = 0;
    PHPSCI_MAIN_MEM_STACK.capacity = 1;
    PHPSCI_MAIN_MEM_STACK.last_deleted_uuid = NULL;
    PHPSCI_MAIN_MEM_STACK.bsize = size;
    // Allocate first CArray struct to buffer
    if((PHPSCI_MAIN_MEM_STACK.buffer = (struct CArray*)emalloc(2 * sizeof(struct CArray))) == NULL){
        php_printf("[MEMORY STACK] MALLOC FAILED\n");
    }
}

/**
 * Grow MemoryStack buffer to new_capacity.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param new_capacity int New capacity for MemoryStack (Buffer)
 *
 * @todo Check if this won't fck everything as the computing requirements grow up
 */
void buffer_to_capacity(int new_capacity, size_t size) {
    PHPSCI_MAIN_MEM_STACK.bsize += size;
    if((PHPSCI_MAIN_MEM_STACK.buffer = (struct CArray*)erealloc(PHPSCI_MAIN_MEM_STACK.buffer, (new_capacity * sizeof(CArray) + sizeof(CArray))))==NULL) {
        php_printf("[MEMORY STACK] REALLOC FAILED\n");
    }

    // Set new capacity to MemoryStack
    PHPSCI_MAIN_MEM_STACK.capacity = new_capacity;
}


/**
 * Add CArray to MemoryStack (Buffer) and retrieve MemoryPointer
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param array CArray CArray to add into the stack
 * @param size  size_t Size of CArray in bytes
 */
void add_to_stack(MemoryPointer * ptr, struct CArray array, size_t size) {
    // If current MemoryStack buffer is empty, initialize it
    if(PHPSCI_MAIN_MEM_STACK.buffer == UNINITIALIZED) {
        stack_init(size);
    }

    // If current capacity is smaller them the requested capacity, grow the MemoryStack
    if((PHPSCI_MAIN_MEM_STACK.size+1) > PHPSCI_MAIN_MEM_STACK.capacity) {
        buffer_to_capacity((PHPSCI_MAIN_MEM_STACK.capacity+1),size);
    }
    // Copy CArray to the MemoryStack Buffer, check if there are preallocated CArray pointers empty
    if(array.array0d != NULL) {
        PHPSCI_MAIN_MEM_STACK.buffer[PHPSCI_MAIN_MEM_STACK.size].array0d = array.array0d;
    }
    if(array.array1d != NULL) {
        PHPSCI_MAIN_MEM_STACK.buffer[PHPSCI_MAIN_MEM_STACK.size].array1d = array.array1d;
    }
    if(array.array2d != NULL) {
        PHPSCI_MAIN_MEM_STACK.buffer[PHPSCI_MAIN_MEM_STACK.size].array2d = array.array2d;
    }

    // Associate CArray unique id
    ptr->uuid = (int)PHPSCI_MAIN_MEM_STACK.size;
    // Set new size for MemoryStack
    PHPSCI_MAIN_MEM_STACK.size++;
}

