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

#include "transformations.h"
#include "../phpsci.h"
#include "../kernel/carray/carray.h"

/**
 * Transpose a CArray 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void
transpose(MemoryPointer * new_ptr, MemoryPointer * target_ptr, int x, int y) {
    int i, j;
    carray_init(y, x, new_ptr);
    CArray new_arr = ptr_to_carray(new_ptr);
    CArray target_arr = ptr_to_carray(target_ptr);
    for(i = 0; i < x; i++) {
        for(j = 0; j < y; j++) {
            new_arr.array2d[(i * y) + j] = target_arr.array2d[(j * x) + i];
        }
    }
}

/**
 * Return a copy of the CArray collapsed into 1D.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param new_ptr
 * @param target_ptr
 */
void
flatten(MemoryPointer * new_ptr, MemoryPointer * target_ptr) {
    int iterator_x, iterator_y, total_iterations = 0;
    CArray target_arr = ptr_to_carray(target_ptr);
    if(IS_0D(target_ptr->x, target_ptr->y)) {
        carray_init1d(1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array1d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 0;
        return;
    }
    if(IS_1D(target_ptr->x, target_ptr->y)) {
        carray_init1d(target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array1d, target_arr.array1d, target_ptr->x * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = 0;
        return;
    }
    if(IS_2D(target_ptr->x, target_ptr->y)) {
        carray_init1d((target_ptr->x * target_ptr->y), new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        for(iterator_x = 0; iterator_x < target_ptr->x; ++iterator_x) {
            for(iterator_y = 0; iterator_y < target_ptr->y; ++ iterator_y) {
                new_arr.array1d[total_iterations] = target_arr.array2d[(iterator_y * target_ptr->x) + iterator_x];
                ++total_iterations;
            }
        }
        new_ptr->x = total_iterations;
        new_ptr->y = 0;
        return;
    }
}

/**
 * Convert inputs to matrices with at least one dimension.
 * Matrices with more dimensions are ignored.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param new_ptr
 * @param target_ptr
 */
void
atleast_1d(MemoryPointer * new_ptr, MemoryPointer * target_ptr) {
    CArray target_arr = ptr_to_carray(target_ptr);
    if(IS_0D(target_ptr->x, target_ptr->y)) {
        carray_init1d(1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array1d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 0;
        return;
    }
    if(IS_1D(target_ptr->x, target_ptr->y)) {
        carray_init1d(target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array1d, target_arr.array1d, target_ptr->x * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = 0;
        return;
    }
    if(IS_2D(target_ptr->x, target_ptr->y)) {
        carray_init(target_ptr->x, target_ptr->y, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array2d, target_arr.array2d, (target_ptr->x * target_ptr->y) * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = target_ptr->y;
        return;
    }
}

/**
 * Convert inputs to matrices with at least two dimension.
 * Matrices with more dimensions are ignored.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param new_ptr
 * @param target_ptr
 */
void
atleast_2d(MemoryPointer * new_ptr, MemoryPointer * target_ptr) {
    int iterator_x;
    CArray target_arr = ptr_to_carray(target_ptr);
    if(IS_0D(target_ptr->x, target_ptr->y)) {
        carray_init(1, 1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array2d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 1;
        return;
    }
    if(IS_1D(target_ptr->x, target_ptr->y)) {
        carray_init(1, target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        for(iterator_x = 0; iterator_x < target_ptr->x; ++iterator_x) {
            new_arr.array2d[(iterator_x * 1) + 0] = target_arr.array1d[iterator_x];
        }
        new_ptr->x = 1;
        new_ptr->y = target_ptr->x;
        return;
    }
    if(IS_2D(target_ptr->x, target_ptr->y)) {
        carray_init(target_ptr->x, target_ptr->y, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array2d, target_arr.array2d, (target_ptr->x * target_ptr->y) * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = target_ptr->y;
        return;
    }
}

