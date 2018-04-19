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
#include "basic_operations.h"
#include "../phpsci.h"
#include "../kernel/carray.h"
#include "initializers.h"

/**
 * Sum of CArray elements.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param x
 * @param y
 * @param axis
 */
void
exponential(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y)
{
    int i, j;
    CArray new_arr = ptr_to_carray(ptr);
    // If 2D Matrix
    if(IS_2D(x, y)) {
        carray_init(x, y, target_ptr);
        // Transform MemoryPointer to CArray
        CArray rtn_arr = ptr_to_carray(target_ptr);
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array2d[(j * x) + i] = exp(new_arr.array2d[(j * x) + i]);
            }
        }
        return;
    }
    // If 1D Matrix
    if(IS_1D(x, y)) {
        carray_init1d(x, target_ptr);
        // Transform MemoryPointer to CArray
        CArray rtn_arr = ptr_to_carray(target_ptr);
        for(i = 0; i < x; ++i) {
            rtn_arr.array1d[i] = exp(new_arr.array1d[i]);
        }

        return;
    }
    // If Scalar
    if(IS_0D(x, y)) {
        carray_init0d(target_ptr);
        // Transform MemoryPointer to CArray
        CArray rtn_arr = ptr_to_carray(target_ptr);
        for(i = 0; i < x; ++i) {
            rtn_arr.array0d[0] = exp(new_arr.array0d[0]);
        }

        return;
    }
}

/**
 * Sum of CArray elements.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param x
 * @param y
 * @param axis
 */
void
sum_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y) {
    int i, j;
    // Initialize CArray
    carray_init0d(target_ptr);
    // Transform MemoryPointer to CArray
    CArray rtn_arr = ptr_to_carray(target_ptr);
    CArray new_arr = ptr_to_carray(ptr);
    // Initialize return to 0
    rtn_arr.array0d[0] = 0;
    // If 2D Matrix
    if(IS_2D(x, y)) {
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array0d[0] += new_arr.array2d[(j * x) + i];
            }
        }
        return;
    }
    // If 1D Matrix
    if(IS_1D(x, y)) {
        for(i = 0; i < x; ++i) {
            rtn_arr.array0d[0] += new_arr.array1d[i];
        }
        
        return;
    }
}

/**
 * Sum of CArray elements over a given axis.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param x
 * @param y
 * @param axis
 */
void
sum_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis, int * size_x, int * size_y) {
    int i, j;
    if(axis == 0) {
        *size_x = y;
        *size_y = 0;
        carray_init1d(y, target_ptr);
        CArray rtn_arr = ptr_to_carray(target_ptr);
        CArray arr = ptr_to_carray(ptr);
        zeros1d(&rtn_arr, y);
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array1d[j] += arr.array2d[(j * x) + i];
            }
        }
        return;
    }
    if(axis == 1) {
        *size_x = x;
        *size_y = 0;
        carray_init1d(x, target_ptr);
        CArray rtn_arr = ptr_to_carray(target_ptr);
        CArray arr = ptr_to_carray(ptr);
        zeros1d(&rtn_arr, y);
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array1d[i] += arr.array2d[(j * x) + i];
            }
        }
        return;
    }
}
