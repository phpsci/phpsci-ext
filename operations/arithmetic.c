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

#include "arithmetic.h"
#include "../phpsci.h"
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"
#include "initializers.h"
#include "zend_exceptions.h"

/**
 * Subtract arguments element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param x_a
 * @param y_a
 * @param ptr_b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void subtract(MemoryPointer * ptr_a,  MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 0) {
        subtract_carray_0d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 1) {
        subtract_carray_1d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 0) {
        subtract_carray_1d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 1) {
        subtract_carray_1d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 2) {
        subtract_carray_2d(ptr_b, ptr_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 0) {
        subtract_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 2) {
        subtract_carray_2d(ptr_b, ptr_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 1) {
        subtract_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 2) {
        subtract_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
}

/**
 * Subtract arguments element-wise if both CArrays provided are 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void subtract_carray_0d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    *size_x = 0;
    *size_y = 0;
    CArray a = ptr_to_carray(ptr_a);
    CArray b = ptr_to_carray(ptr_b);
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = carray_a.array0d[0] - carray_b.array0d[0];
    return;
}

/**
 * Subtract arguments element-wise if one or both CArrays provided are 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void subtract_carray_1d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    if(GET_DIM(ptr_b) == 0) {
        carray_init1d(ptr_a->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = 0;
        for(i = 0; i < ptr_a->x; ++i) {
            rtn_arr.array1d[i] = carray_a.array1d[i] - carray_b.array0d[0];
        }
        return;
    }
    if(GET_DIM(ptr_b) == 1 && GET_DIM(ptr_a) == 0) {
        carray_init1d(ptr_b->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_b->x;
        *size_y = 0;
        for(i = 0; i < ptr_b->x; ++i) {
            rtn_arr.array1d[i] = carray_a.array0d[0] - carray_b.array1d[i];
        }
        return;
    }
    carray_init1d(ptr_a->x, rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    for(i = 0; i < ptr_a->x; ++i) {
        rtn_arr.array1d[i] = carray_a.array1d[i] - carray_b.array1d[i];
    }
    *size_x = ptr_a->x;
    *size_y = 0;
    return;
}

/**
 * Subtract arguments element-wise if one or both CArrays provided are 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 * @param size_x
 * @param size_y
 */
void subtract_carray_2d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    if(GET_DIM(ptr_b) == 0) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] - carray_b.array0d[0];
            }
        }
        return;
    }
    if(GET_DIM(ptr_b) == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] - carray_b.array1d[j];
            }
        }
        return;
    }
    if(ptr_a->x == ptr_b->x) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] - carray_b.array2d[(j * ptr_a->x) + i];
            }
        }
        return;
    }
    if(ptr_b->x < ptr_a->x && ptr_b->x == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] - carray_b.array2d[(j * ptr_b->x) + 0];
            }
        }
        return;
    }
    PHPSCI_THROW("Could not broadcast operands.", 4563);
    return;
}

/**
 * Add arguments element-wise.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param x_a
 * @param y_a
 * @param ptr_b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void add(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 0) {
        add_carray_0d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 1) {
        add_carray_1d(ptr_b, ptr_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 0) {
        add_carray_1d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 1) {
        add_carray_1d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 0 && GET_DIM(ptr_b) == 2) {
        add_carray_2d(ptr_b, ptr_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 0) {
        add_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 1 && GET_DIM(ptr_b) == 2) {
        add_carray_2d(ptr_b, ptr_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 1) {
        add_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(ptr_a) == 2 && GET_DIM(ptr_b) == 2) {
        add_carray_2d(ptr_a, ptr_b, rtn_ptr, size_x, size_y);
        return;
    }
}

/**
 * Add arguments element-wise if both CArrays provided are 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void add_carray_0d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    *size_x = 0;
    *size_y = 0;
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = carray_a.array0d[0] + carray_b.array0d[0];
    return;
}

/**
 * Add arguments element-wise if one or both CArrays provided are 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 */
void add_carray_1d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    if(GET_DIM(ptr_b) == 0) {
        carray_init1d(ptr_a->x, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = 0;
        for(i = 0; i < ptr_a->x; ++i) {
            rtn_arr.array1d[i] = carray_a.array1d[i] + carray_b.array0d[0];
        }
        return;
    }
    carray_init1d(ptr_a->x, rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    for(i = 0; i < ptr_a->x; ++i) {
        rtn_arr.array1d[i] = carray_a.array1d[i] + carray_b.array1d[i];
    }
    *size_x = ptr_a->x;
    *size_y = 0;
    return;
}

/**
 * Add arguments element-wise if one or both CArrays provided are 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param a
 * @param x_a
 * @param y_a
 * @param b
 * @param x_b
 * @param y_b
 * @param rtn_ptr
 * @param size_x
 * @param size_y
 */
void add_carray_2d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    int i, j;
    if(GET_DIM(ptr_b) == 0) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] + carray_b.array0d[0];
            }
        }
        return;
    }
    if(GET_DIM(ptr_b) == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] + carray_b.array1d[j];
            }
        }
        return;
    }
    if(ptr_a->x == ptr_b->x) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] + carray_b.array2d[(j * ptr_a->x) + i];
            }
        }
        return;
    }
    if(ptr_b->x < ptr_a->x && ptr_b->x == 1) {
        carray_init(ptr_a->x, ptr_a->y, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = ptr_a->x;
        *size_y = ptr_a->y;
        for(i = 0; i < ptr_a->x; ++i) {
            for(j = 0; j < ptr_a->y; ++j) {
                rtn_arr.array2d[(j * ptr_a->x) + i] = carray_a.array2d[(j * ptr_a->x) + i] + carray_b.array2d[(j * ptr_b->x) + 0];
            }
        }
        return;
    }
    PHPSCI_THROW("Could not broadcast operands.", 4563);
    return;
}
