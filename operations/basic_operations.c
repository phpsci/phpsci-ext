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
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"
#include "initializers.h"
#include "statistics.h"
#include "cblas.h"
#include "lapacke.h"

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
sum_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y)
{
    int i, j;
    // Initialize CArray
    carray_init0d(target_ptr);
    // Transform MemoryPointer to CArray
    CArray rtn_arr = ptr_to_carray(target_ptr);
    CArray new_arr = ptr_to_carray(ptr);
    // Initialize return to 0
    rtn_arr.array0d[0] = 0;
    // If 2D Matrix
    if(IS_2D(ptr)) {
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array0d[0] += new_arr.array2d[(j * x) + i];
            }
        }
        return;
    }
    // If 1D Matrix
    if(IS_1D(ptr)) {
        double y_1 = 1.0;
        rtn_arr.array0d[0] = cblas_ddot(x, new_arr.array1d, 1, &y_1, 0);
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
sum_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis, int * size_x, int * size_y)
{
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

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param target_ptr
 */
void
negative(MemoryPointer * ptr, MemoryPointer * target_ptr)
{
    int i = 0, j = 0;
    CArray input = ptr_to_carray(ptr);
    if(IS_0D(ptr)) {
        carray_init0d(target_ptr);
        CArray target = ptr_to_carray(target_ptr);
        target.array0d[0] = input.array0d[0] * -1;
        return;
    }
    if(IS_1D(ptr)) {
        carray_init1d(ptr->x, target_ptr);
        CArray target = ptr_to_carray(target_ptr);
        for(i = 0; i < ptr->x; i++) {
            target.array1d[i] = input.array1d[i] * -1;
        }
    }
    if(IS_2D(ptr)) {
        carray_init(ptr->x, ptr->y, target_ptr);
        CArray target = ptr_to_carray(target_ptr);
        for(i = 0; i < ptr->x; i++) {
            for(j = 0; j < ptr->y; j++) {
                target.array2d[(j * ptr->x) + i] = input.array2d[(j * ptr->x) + i] * -1;
            }
        }
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a 0D
 * @param ptr_b ND
 * @param rtn_ptr
 */
void
multiply_0d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    int i = 0, j = 0;
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    CArray rtn_carray;
    if(IS_0D(ptr_b)) {
        carray_init0d(rtn_ptr);
        rtn_carray = ptr_to_carray(rtn_ptr);
        rtn_carray.array0d[0] = carray_a.array0d[0] * carray_b.array0d[0];
        return;
    }
    if(IS_1D(ptr_b)) {
        carray_init1d(ptr_b->x, rtn_ptr);
        rtn_carray = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_b->x; i++) {
            rtn_carray.array1d[i] = carray_a.array0d[0] * carray_b.array1d[i];
        }
    }
    if(IS_2D(ptr_b)) {
        carray_init(ptr_b->x, ptr_b->y, rtn_ptr);
        rtn_carray = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_b->x; i++) {
            for(j = 0; j < ptr_b->y; j++) {
                rtn_carray.array2d[(j * ptr_b->x) + i] = carray_a.array0d[0] / carray_b.array2d[(j * ptr_b->x) + i];
            }
        }
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
multiply_1d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    int i = 0;
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    CArray rtn_carray;
    if(IS_1D(ptr_b)) {
        carray_init1d(ptr_b->x, rtn_ptr);
        rtn_carray = ptr_to_carray(rtn_ptr);
        for(i; i < ptr_b->x; i++) {
            rtn_carray.array1d[i] = carray_a.array1d[i] * carray_b.array1d[i];
        }
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
multiply_2d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    int i = 0, j = 0;
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    CArray rtn_carray;
    if(IS_2D(ptr_b)) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        rtn_carray = ptr_to_carray(rtn_ptr);
        for(i; i < ptr_a->x; i++) {
            for(j; j < ptr_a->y; j++) {
                rtn_carray.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] * carray_b.array2d[(j * ptr_a->x) + i];
            }
        }
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
multiply(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    if(IS_0D(ptr_a) || IS_0D(ptr_b)) {
        if(IS_0D(ptr_a)) {
            multiply_0d(ptr_a, ptr_b, rtn_ptr);
        }
        if(IS_0D(ptr_b)) {
            multiply_0d(ptr_b, ptr_a, rtn_ptr);
        }
        return;
    }

    if(IS_1D(ptr_a) || IS_1D(ptr_b)) {
        if(IS_1D(ptr_a)) {
            multiply_1d(ptr_a, ptr_b, rtn_ptr);
        }
        if(IS_1D(ptr_b)) {
            multiply_1d(ptr_b, ptr_a, rtn_ptr);
        }
        return;
    }

    if(IS_2D(ptr_a) || IS_2D(ptr_b)) {
        if(IS_2D(ptr_a)) {
            multiply_2d(ptr_a, ptr_b, rtn_ptr);
        }
        if(IS_2D(ptr_b)) {
            multiply_2d(ptr_b, ptr_a, rtn_ptr);
        }
        return;
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a 0D
 * @param ptr_b ND
 * @param rtn_ptr
 */
void
divide_0d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    CArray rtn_carray;
    if(IS_0D(ptr_a)) {
        if(IS_0D(ptr_b)) {
            carray_init0d(rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            rtn_carray.array0d[0] = carray_a.array0d[0] / carray_b.array0d[0];
            return;
        }
        if(IS_1D(ptr_b)) {
            carray_init1d(ptr_b->x, rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            for(i; i < ptr_b->x; i++) {
                rtn_carray.array1d[i] = carray_a.array0d[0] / carray_b.array1d[i];
            }
        }
        if(IS_2D(ptr_b)) {
            carray_init(ptr_b->x, ptr_b->y, rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            for(i = 0; i < ptr_b->x; i++) {
                for(j = 0; j < ptr_b->y; j++) {
                    rtn_carray.array2d[(j * ptr_b->x) + i] = carray_a.array0d[0] / carray_b.array2d[(j * ptr_b->x) + i];
                }
            }
        }
        return;
    }
    if(IS_0D(ptr_b)) {
        if(IS_0D(ptr_a)) {
            carray_init0d(rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            rtn_carray.array0d[0] = carray_a.array0d[0] / carray_b.array0d[0];
            return;
        }
        if(IS_1D(ptr_a)) {
            carray_init1d(ptr_a->x, rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            for(i; i < ptr_a->x; i++) {
                rtn_carray.array1d[i] = carray_a.array1d[i] / carray_b.array0d[0];
            }
        }
        if(IS_2D(ptr_a)) {
            carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
            rtn_carray = ptr_to_carray(rtn_ptr);
            for(i = 0; i < ptr_a->x; i++) {
                for(j = 0; j < ptr_a->y; j++) {
                    rtn_carray.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] / carray_b.array0d[0];
                }
            }
        }
    }
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
divide_1d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    // @todo IMPLEMENT DIVIDE 1D
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
divide_2d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    if(GET_DIM(ptr_b) == 0) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] / carray_b.array0d[0];
            }
        }
        return;
    }
    if(GET_DIM(ptr_b) == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] / carray_b.array1d[j];
            }
        }
        return;
    }
    if(ptr_a->x == ptr_b->x) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] / carray_b.array2d[(j * ptr_a->x) + i];
            }
        }
        return;
    }
    if(ptr_b->x < ptr_a->x && ptr_b->x == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] / carray_b.array2d[(j * ptr_b->x) + 0];
            }
        }
        return;
    }
}

/**
 *
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
divide(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    if(IS_2D(ptr_a) || IS_2D(ptr_b)) {
        if(IS_2D(ptr_a)) {
            divide_2d(ptr_a, ptr_b, rtn_ptr);
            return;
        }
        if(IS_2D(ptr_b)) {
            divide_2d(ptr_b, ptr_a, rtn_ptr);
            return;
        }
    }
    if(IS_0D(ptr_a) || IS_0D(ptr_b)) {
        divide_0d(ptr_a, ptr_b, rtn_ptr);
        return;
    }

    if(IS_1D(ptr_a) || IS_1D(ptr_b)) {
        divide_1d(ptr_a, ptr_b, rtn_ptr);
        return;
    }
}

/**
 * @param ptr_a
 */
void
square(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray target = ptr_to_carray(ptr_a);
    CArray rtn;
    zeros(rtn_ptr, ptr_a->x, ptr_a->y);
    rtn = ptr_to_carray(rtn_ptr);
    if(IS_1D(ptr_a)) {
        for(i = 0; i < ptr_a->x; i++) {
            rtn.array1d[i] = target.array1d[i] * target.array1d[i];
        }
    }
    if(IS_2D(ptr_a)) {
        for(i = 0; i < ptr_a->x; i++) {
            for (j = 0; j < ptr_a->y; j++) {
                rtn.array2d[(j * ptr_a->x) + i] = target.array2d[(j * ptr_a->x) + i] * target.array2d[(j * ptr_a->x) + i];
            }
        }
    }
}

/**
 * @param ptr_a
 * @param rtn_ptr
 */
void
absolute(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    int i, j;
    CArray target = ptr_to_carray(ptr_a);
    CArray rtn;
    if(IS_1D(ptr_a)) {
        carray_init1d(ptr_a->x, rtn_ptr);
        rtn = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; i++) {
            rtn.array1d[i] = fabs(target.array1d[i]);
        }
    }
    if(IS_2D(ptr_a)) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        rtn = ptr_to_carray(rtn_ptr);
        for(i = 0; i < ptr_a->x; i++) {
            for (j = 0; j < ptr_a->y; j++) {
                rtn.array2d[(j * rtn_ptr->x) + i] = fabs(target.array2d[(j * ptr_a->x) + i]);
            }
        }
    }
}

/**
 * @param ptr_a
 * @return
 */
int
all(MemoryPointer * ptr_a, int axis)
{
    int i;
    CArray array_a = ptr_to_carray(ptr_a);
    MemoryPointer temp;
    if(IS_1D(ptr_a)) {
        mean(ptr_a, &temp, INT_MAX);
        CArray temp_ca = ptr_to_carray(&temp);
        for(i = 0; i < ptr_a->x; i++) {
            if(array_a.array1d[i] != temp_ca.array0d[0]) {
                return 0;
            }
        }
        return 1;
    }
}