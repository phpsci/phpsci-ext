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

#include "statistics.h"
#include "../phpsci.h"
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"
#include "initializers.h"
#include "arithmetic.h"
#include "basic_operations.h"


/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
double amin(MemoryPointer * ptr_a)
{
    double lowest;
    CArray input_carray = ptr_to_carray(ptr_a);
    if (IS_0D(ptr_a)) {
        lowest = input_carray.array0d[0];
    }
    if (IS_1D(ptr_a)) {
        for(int i = 0; i < ptr_a->x; i++) {
            if(!lowest) {
                lowest = input_carray.array1d[i];
            }
            if(lowest > input_carray.array1d[i]) {
                lowest = input_carray.array1d[i];
            }
        }
    }
    if (IS_2D(ptr_a)) {
        for(int i = 0; i < ptr_a->x; i++) {
            for(int j = 0; j < ptr_a->y; j++) {
                if(!lowest) {
                    lowest = input_carray.array2d[(j * ptr_a->x) + i];
                }
                if (lowest > input_carray.array2d[(j * ptr_a->x) + i]) {
                    lowest = input_carray.array2d[(j * ptr_a->x) + i];
                }
            }
        }
    }
    return lowest;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
double amax(MemoryPointer * ptr_a)
{
    double lowest;
    CArray input_carray = ptr_to_carray(ptr_a);
    if (IS_0D(ptr_a)) {
        lowest = input_carray.array0d[0];
    }
    if (IS_1D(ptr_a)) {
        for(int i = 0; i < ptr_a->x; i++) {
            if(!lowest) {
                lowest = input_carray.array1d[i];
            }
            if(lowest < input_carray.array1d[i]) {
                lowest = input_carray.array1d[i];
            }
        }
    }
    if (IS_2D(ptr_a)) {
        for(int i = 0; i < ptr_a->x; i++) {
            for(int j = 0; j < ptr_a->y; j++) {
                if(!lowest) {
                    lowest = input_carray.array2d[(j * ptr_a->x) + i];
                }
                if (lowest < input_carray.array2d[(j * ptr_a->x) + i]) {
                    lowest = input_carray.array2d[(j * ptr_a->x) + i];
                }
            }
        }
    }
    return lowest;
}

/**
 * Calculate the arithmetic mean over given axis
 *
 * @param ptr
 */
void
mean(MemoryPointer * ptr, MemoryPointer * rtn_ptr, int axis)
{
    CArray target = ptr_to_carray(ptr);
    CArray rtn;
    int totalsum = 0;
    if(IS_1D(ptr)) {
        for(int i; i < ptr->x; i++) {
            totalsum += target.array1d[i];
        }
        carray_init0d(rtn_ptr);
        rtn = ptr_to_carray(rtn_ptr);
        rtn.array0d[0] = totalsum / ptr->x;
    }
    if(IS_2D(ptr)) {
        // 2D MATRIX WITH AXIS 0
        if(axis == INT_MAX) {
            for(int i; i < ptr->x; i++) {
                for(int j = 0; j < ptr->y; j++) {
                    totalsum += target.array2d[(j * ptr->x) + i];
                }
            }
            carray_init0d(rtn_ptr);
            rtn = ptr_to_carray(rtn_ptr);
            rtn.array0d[0] = totalsum / ((double)(ptr->x * ptr->y));
        }
        if(axis == 0) {
            zeros(rtn_ptr, ptr->y, 0);
            rtn = ptr_to_carray(rtn_ptr);
            for(int i; i < ptr->x; i++) {
                for(int j = 0; j < ptr->y; j++) {
                    rtn.array1d[j] += target.array2d[(j * ptr->x) + i];
                }
            }
            for(int j = 0; j < ptr->y; j++) {
                rtn.array1d[j] = rtn.array1d[j] / ((double)(ptr->x));
            }
        }
    }
}


/**
 * Compute the variance along the axis
 *
 * @param ptr
 * @param rtn_ptr
 * @param axis
 */
void
var(MemoryPointer * ptr, MemoryPointer * rtn_ptr, int axis)
{
    CArray target = ptr_to_carray(ptr);
    CArray rtn;
    MemoryPointer mean_temp_ptr, subtract_tmp_ptr, square_tmp_ptr, absolute_tmp_ptr;
    if(IS_2D(ptr)) {
        if(axis == 0) {
            mean(ptr, &mean_temp_ptr, 0);
            subtract(ptr, &mean_temp_ptr, &subtract_tmp_ptr, &(subtract_tmp_ptr.x), &(subtract_tmp_ptr.y));
            square(&subtract_tmp_ptr, &square_tmp_ptr);
            absolute(&square_tmp_ptr, &absolute_tmp_ptr);
            mean(&absolute_tmp_ptr, rtn_ptr, 0);
        }
        destroy_carray(&mean_temp_ptr);
        destroy_carray(&subtract_tmp_ptr);
        destroy_carray(&square_tmp_ptr);
        destroy_carray(&absolute_tmp_ptr);
    }
}