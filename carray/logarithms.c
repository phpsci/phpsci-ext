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
#include "logarithms.h"
#include "../phpsci.h"
#include "../kernel/memory_manager.h"

/**
 * Natural logarithm, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
natural_log(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = log(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = log(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = log(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Base 10 logarithm, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
base10_log(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = log10(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = log10(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = log10(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Base 2 logarithm, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
base2_log(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = log(carray.array0d[0]) / log(2);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = log(carray.array1d[i]) / log(2);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = log(carray.array2d[(j * ptr->x) + i]) / log(2);
            }
        }
        return;
    }
}
