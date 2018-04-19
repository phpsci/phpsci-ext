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

#include "exponents.h"
#include "../kernel/carray.h"
#include "../kernel/memory_manager.h"


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
