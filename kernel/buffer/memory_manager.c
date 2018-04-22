/*
  +----------------------------------------------------------------------+
  | PHPSci CArray                                                        |
  +----------------------------------------------------------------------+
  | Copyright (c) 2018 PHPSci Team                                       |
  +----------------------------------------------------------------------+
  | Licensed under the Apache License, Version 2.0 (the "License");      |
  | you may not use this file except in compliance with the License.     |
  | You may obtain a copy of the License at                              |
  |                                                                      |
  |     http://www.apache.org/licenses/LICENSE-2.0                       |
  |                                                                      |
  | Unless required by applicable law or agreed to in writing, software  |
  | distributed under the License is distributed on an "AS IS" BASIS,    |
  | WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or      |
  | implied.                                                             |
  | See the License for the specific language governing permissions and  |
  | limitations under the License.                                       |
  +----------------------------------------------------------------------+
  | Authors: Henrique Borba <henrique.borba.dev@gmail.com>               |
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
struct MemoryStack PHPSCI_MAIN_MEM_STACK = {NULL,UNINITIALIZED,UNINITIALIZED, UNINITIALIZED};

/**
 * Initialize MemoryStack Buffer
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @todo Same from buffer_to_capacity
 */
void stack_init(size_t size) {
    PHPSCI_MAIN_MEM_STACK.size = 0;
    PHPSCI_MAIN_MEM_STACK.capacity = 1;
    PHPSCI_MAIN_MEM_STACK.last_deleted_uuid = UNINITIALIZED;
    PHPSCI_MAIN_MEM_STACK.bsize = size;
    // Allocate first CArray struct to buffer
    PHPSCI_MAIN_MEM_STACK.buffer = (struct CArray*)emalloc(sizeof(struct CArray));
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
    PHPSCI_MAIN_MEM_STACK.buffer = (struct CArray*)erealloc(PHPSCI_MAIN_MEM_STACK.buffer, (new_capacity * sizeof(CArray)));
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
    if(PHPSCI_MAIN_MEM_STACK.buffer == NULL) {
        stack_init(size);
    }

    // If current capacity is smaller them the requested capacity, grow the MemoryStack
    if((PHPSCI_MAIN_MEM_STACK.size+1) > PHPSCI_MAIN_MEM_STACK.capacity) {
        buffer_to_capacity((PHPSCI_MAIN_MEM_STACK.capacity+1),size);
    }

    PHPSCI_MAIN_MEM_STACK.buffer[PHPSCI_MAIN_MEM_STACK.size] = array;

    // Associate CArray unique id
    ptr->uuid = (int)PHPSCI_MAIN_MEM_STACK.size;
    // Set new size for MemoryStack
    PHPSCI_MAIN_MEM_STACK.size++;
}

