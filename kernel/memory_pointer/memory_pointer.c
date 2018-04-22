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
#include "memory_pointer.h"
#include "../carray/carray.h"
#include "../buffer/memory_manager.h"
#include "utils.h"

/**
 * Copy CArray from MemoryPointer A (ptr_a) to MemoryPointer B (ptr_b)
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 */
void
COPY_PTR(MemoryPointer * ptr_a, MemoryPointer * ptr_b)
{
    CArray to_array;
    CArray from_array = ptr_to_carray(ptr_a);
    if(IS_0D(ptr_a)) {
        carray_init0d(ptr_b);
        to_array = ptr_to_carray(ptr_b);
        memcpy(to_array.array0d, from_array.array0d, sizeof(double));
        return;
    }
    if(IS_1D(ptr_a)) {
        carray_init1d(ptr_a->x, ptr_b);
        to_array = ptr_to_carray(ptr_b);
        memcpy(to_array.array1d, from_array.array1d, (ptr_a->x * sizeof(double)));
        return;
    }
    if(IS_2D(ptr_a)) {
        carray_init(ptr_a->x, ptr_a->y, ptr_b);
        to_array = ptr_to_carray(ptr_b);
        memcpy(to_array.array2d, from_array.array2d, (ptr_a->x * ptr_a->y * sizeof(double)));
        return;
    }
    ptr_b->x = ptr_a->x;
    ptr_b->y = ptr_a->y;
}