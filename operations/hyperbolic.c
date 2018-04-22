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

#include "hyperbolic.h"
#include "../phpsci.h"
#include "../kernel/buffer/memory_manager.h"
#include "../kernel/carray/carray.h"
#include "math.h"

/**
 * Hyperbolic sine, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param rtn_ptr
 */
void
hyperbolic_tanh(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = tanh(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = tanh(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = tanh(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Hyperbolic cosine, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param rtn_ptr
 */
void
hyperbolic_cosh(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = cosh(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = cosh(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = cosh(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Compute hyperbolic tangent element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param rtn_ptr
 */
void
hyperbolic_sinh(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = sinh(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = sinh(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = sinh(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

