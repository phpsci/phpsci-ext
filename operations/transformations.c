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
#include "../kernel/memory_pointer/utils.h"
#include "../kernel/memory_pointer/memory_pointer.h"
#include "../operations/initializers.h"

/**
 * Transpose a CArray 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void
transpose(MemoryPointer * new_ptr, MemoryPointer * target_ptr) {
    int i, j;
    carray_init(target_ptr->y, target_ptr->x, new_ptr);
    CArray new_arr = ptr_to_carray(new_ptr);
    CArray target_arr = ptr_to_carray(target_ptr);
    for(i = 0; i < target_ptr->x; i++) {
        for(j = 0; j < target_ptr->y; j++) {
            new_arr.array2d[(i * target_ptr->y) + j] = target_arr.array2d[(j * target_ptr->x) + i];
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
    if(IS_0D(target_ptr)) {
        carray_init1d(1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array1d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 0;
        return;
    }
    if(IS_1D(target_ptr)) {
        carray_init1d(target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array1d, target_arr.array1d, target_ptr->x * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = 0;
        return;
    }
    if(IS_2D(target_ptr)) {
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
    if(IS_0D(target_ptr)) {
        carray_init1d(1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array1d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 0;
        return;
    }
    if(IS_1D(target_ptr)) {
        carray_init1d(target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array1d, target_arr.array1d, target_ptr->x * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = 0;
        return;
    }
    if(IS_2D(target_ptr)) {
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
    if(IS_0D(target_ptr)) {
        carray_init(1, 1, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        new_arr.array2d[0] = target_arr.array0d[0];
        new_ptr->x = 1;
        new_ptr->y = 1;
        return;
    }
    if(IS_1D(target_ptr)) {
        carray_init(1, target_ptr->x, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        for(iterator_x = 0; iterator_x < target_ptr->x; ++iterator_x) {
            new_arr.array2d[(iterator_x * 1) + 0] = target_arr.array1d[iterator_x];
        }
        new_ptr->x = 1;
        new_ptr->y = target_ptr->x;
        return;
    }
    if(IS_2D(target_ptr)) {
        carray_init(target_ptr->x, target_ptr->y, new_ptr);
        CArray new_arr = ptr_to_carray(new_ptr);
        memcpy(new_arr.array2d, target_arr.array2d, (target_ptr->x * target_ptr->y) * sizeof(double));
        new_ptr->x = target_ptr->x;
        new_ptr->y = target_ptr->y;
        return;
    }
}

int cmpfunc (const void * a, const void * b)
{
    return ( *(double*)a - *(double*)b );
}

/**
 * Find the unique elements of an array.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param new_ptr
 * @param target_ptr
 */
void unique(MemoryPointer * new_ptr, MemoryPointer * target_ptr) {
    int i, j, k = 0, size;
    MemoryPointer tmp_ptr, sorted_target;
    int total_unique = 0;
    if(IS_1D(target_ptr)) {
        COPY_PTR(target_ptr, &sorted_target);
        carray_init1d(target_ptr->x, &tmp_ptr);
        CArray tmp_carray = ptr_to_carray(&tmp_ptr);
        CArray target_carray = ptr_to_carray(&sorted_target);
        qsort(target_carray.array1d, target_ptr->x, sizeof(double), cmpfunc);
        for( i = 0 ; i < target_ptr->x; i++ )
        {
            if(total_unique==0)
                tmp_carray.array1d[total_unique++]=target_carray.array1d[i];
            else
            {
                if(target_carray.array1d[i] == tmp_carray.array1d[total_unique-1]) {
                    continue;
                } else {
                    tmp_carray.array1d[total_unique++] = target_carray.array1d[i];
                }
            }
        }
        carray_init1d(total_unique, new_ptr);
        CArray rtn_carray = ptr_to_carray(new_ptr);
        for(i = 0; i < total_unique; i++) {
            rtn_carray.array1d[i] = tmp_carray.array1d[i];
        }
        destroy_carray(&tmp_ptr);
        destroy_carray(&sorted_target);
    }
}

