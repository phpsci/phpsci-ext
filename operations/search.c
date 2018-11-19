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

#include "search.h"
#include "../phpsci.h"
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"

/**
 * @param ptr_a
 * @param ptr_indices
 */
void
get_indices(MemoryPointer * ptr_a, MemoryPointer * ptr_indices, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray target = ptr_to_carray(ptr_a);
    CArray indices = ptr_to_carray(ptr_indices);
    if(IS_2D(ptr_a)) {
        carray_init(ptr_indices->x, ptr_a->y, rtn_ptr);
        CArray rtn = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_indices->x; i++) {
            for(j = 0; j < ptr_a->y; j++) {
                rtn.array2d[(j * ptr_indices->x) + i] = target.array2d[(j * ptr_a->x) + (int)indices.array1d[i]];
            }
        }
    }
}

/**
 * @param target_ptr
 * @param rtn_ptr
 * @param search
 */
void
search_keys(MemoryPointer * target_ptr, MemoryPointer * rtn_ptr, double search)
{
    int i;
    CArray target = ptr_to_carray(target_ptr);
    int num_allocated = 0;
    if(IS_1D(target_ptr)) {
        int count_found = 0;
        for(i = 0; i < target_ptr->x; i++) {
            if(target.array1d[i] == search) {
                count_found++;
            }
        }

        carray_init1d(count_found, rtn_ptr);
        CArray rtn = ptr_to_carray(rtn_ptr);
        for(i = 0; i < target_ptr->x; i++) {
            if(target.array1d[i] == search) {
                rtn.array1d[num_allocated] = i;
                num_allocated++;
            }
        }
    }
}

/**
 * Returns the indices of the maximum values along an axis.
 * @param target_ptr
 * @param rtn_ptr
 * @param axis
 */
void
argmax(MemoryPointer * target_ptr, MemoryPointer * rtn_ptr, int axis)
{
    int i, j;
    MemoryPointer values_ptr;
    CArray target = ptr_to_carray(target_ptr);
    if(axis == 1) {
        if(IS_2D(target_ptr)) {
            carray_init1d(target_ptr->x, rtn_ptr);
            carray_init1d(target_ptr->x, &values_ptr);
            CArray rtn = ptr_to_carray(rtn_ptr);
            CArray values = ptr_to_carray(&values_ptr);
            for(i = 0; i < target_ptr->x; i++) {
                for(j = 0; j < target_ptr->y; j++) {
                    if(!values.array1d[i]) {
                        values.array1d[i] = -INFINITY;
                    }
                    if(target.array2d[(j * target_ptr->x) + i] > values.array1d[i]) {
                        rtn.array1d[i] = (double)j;
                    }
                }
            }
            destroy_carray(&values_ptr);
        }
    }
}