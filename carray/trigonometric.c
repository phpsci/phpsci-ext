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

#include "trigonometric.h"
#include "../phpsci.h"
#include "../kernel/memory_manager.h"
#include "../kernel/carray.h"
#include "math.h"


/**
 * Compute tangent element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
tan_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = tan(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = tan(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = tan(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * 	Cosine element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
cos_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = cos(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = cos(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = cos(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Trigonometric sine, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
sin_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = sin(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = sin(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = sin(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}


/**
 * Trigonometric inverse tangent, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
arctan_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = atan(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = atan(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = atan(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * 	Trigonometric inverse cosine, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
arccos_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = acos(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = acos(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = acos(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}

/**
 * Inverse sine, element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 */
void
arcsin_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray = ptr_to_carray(ptr);
    if(IS_0D(ptr->x, ptr->y)) {
        carray_init0d(rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        rtn_arr.array0d[0] = asin(carray.array0d[0]);
        return;
    }
    if(IS_1D(ptr->x, ptr->y)) {
        carray_init1d(ptr->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            rtn_arr.array1d[i] = asin(carray.array1d[i]);
        }
        return;
    }
    if(IS_2D(ptr->x, ptr->y)) {
        carray_init(ptr->x, ptr->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr->x; ++i) {
            for(j = 0; j < ptr->y; ++j) {
                rtn_arr.array2d[(j * ptr->x) + i] = asin(carray.array2d[(j * ptr->x) + i]);
            }
        }
        return;
    }
}