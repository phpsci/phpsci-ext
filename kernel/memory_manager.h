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

#ifndef PHPSCI_EXT_MEMORY_MANAGER_H
#define PHPSCI_EXT_MEMORY_MANAGER_H
#define UNINITIALIZED -1

#include "carray.h"

/**
 * MemoryStack : The memory buffer of CArrays
 */
struct MemoryStack {
    CArray * buffer;
    int size;
    int capacity;
    size_t bsize;
    int last_deleted_uuid;
} MemoryStack;

/**
 * PHPSCI_MAIN_MEM_STACK : Global memory stack of CArrays. CArrays are always visible
 * within the runtime.
 *
 * @todo Check if this is bad
 */
extern struct MemoryStack PHPSCI_MAIN_MEM_STACK;


void add_to_stack(MemoryPointer * ptr, CArray array, size_t size);
void buffer_to_capacity(int new_capacity, size_t size);
void stack_init();


#endif //PHPSCI_EXT_MEMORY_MANAGER_H
