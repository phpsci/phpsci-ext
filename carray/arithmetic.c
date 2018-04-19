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
#include "../kernel/carray.h"
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
void subtract(MemoryPointer * ptr_a, int x_a, int y_a, MemoryPointer * ptr_b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray arr_a = ptr_to_carray(ptr_a);
    CArray arr_b = ptr_to_carray(ptr_b);
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 0) {
        subtract_carray_0d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 1) {
        subtract_carray_1d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 0) {
        subtract_carray_1d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 1) {
        subtract_carray_1d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 2) {
        subtract_carray_2d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 0) {
        subtract_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 2) {
        subtract_carray_2d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 1) {
        subtract_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 2) {
        subtract_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
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
void subtract_carray_0d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    *size_x = 0;
    *size_y = 0;
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = a->array0d[0] - b->array0d[0];
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
void subtract_carray_1d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    if(GET_DIM(x_b, y_b) == 0) {
        carray_init1d(x_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = 0;
        for(i = 0; i < x_a; ++i) {
            rtn_arr.array1d[i] = a->array1d[i] - b->array0d[0];
        }
        return;
    }
    carray_init1d(x_a, rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    for(i = 0; i < x_a; ++i) {
        rtn_arr.array1d[i] = a->array1d[i] - b->array1d[i];
    }
    *size_x = x_a;
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
void subtract_carray_2d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    if(GET_DIM(x_b, y_b) == 0) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] - b->array0d[0];
            }
        }
        return;
    }
    if(GET_DIM(x_b, y_b) == 1) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] - b->array1d[j];
            }
        }
        return;
    }
    if(x_a == x_b) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] - b->array2d[(j * x_a) + i];
            }
        }
        return;
    }
    if(x_b < x_a && x_b == 1) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] - b->array2d[(j * x_b) + 0];
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
void add(MemoryPointer * ptr_a, int x_a, int y_a, MemoryPointer * ptr_b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    CArray arr_a = ptr_to_carray(ptr_a);
    CArray arr_b = ptr_to_carray(ptr_b);
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 0) {
        add_carray_0d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 1) {
        add_carray_1d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 0) {
        add_carray_1d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 1) {
        add_carray_1d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 0 && GET_DIM(x_b, y_b) == 2) {
        add_carray_2d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 0) {
        add_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 1 && GET_DIM(x_b, y_b) == 2) {
        add_carray_2d(&arr_b, x_b, y_b, &arr_a, x_a, y_a, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 1) {
        add_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
        return;
    }
    if(GET_DIM(x_a, y_a) == 2 && GET_DIM(x_b, y_b) == 2) {
        add_carray_2d(&arr_a, x_a, y_a, &arr_b, x_b, y_b, rtn_ptr, size_x, size_y);
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
void add_carray_0d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    *size_x = 0;
    *size_y = 0;
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = a->array0d[0] + b->array0d[0];
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
void add_carray_1d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    if(GET_DIM(x_b, y_b) == 0) {
        carray_init1d(x_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = 0;
        for(i = 0; i < x_a; ++i) {
            rtn_arr.array1d[i] = a->array1d[i] + b->array0d[0];
        }
        return;
    }
    carray_init1d(x_a, rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    for(i = 0; i < x_a; ++i) {
        rtn_arr.array1d[i] = a->array1d[i] + b->array1d[i];
    }
    *size_x = x_a;
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
void add_carray_2d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y)
{
    int i, j;
    if(GET_DIM(x_b, y_b) == 0) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] + b->array0d[0];
            }
        }
        return;
    }
    if(GET_DIM(x_b, y_b) == 1) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] + b->array1d[j];
            }
        }
        return;
    }
    if(x_a == x_b) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] + b->array2d[(j * x_a) + i];
            }
        }
        return;
    }
    if(x_b < x_a && x_b == 1) {
        carray_init(x_a, y_a, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        *size_x = x_a;
        *size_y = y_a;
        for(i = 0; i < x_a; ++i) {
            for(j = 0; j < y_a; ++j) {
                rtn_arr.array2d[(j * x_a) + i] = a->array2d[(j * x_a) + i] + b->array2d[(j * x_b) + 0];
            }
        }
        return;
    }
    PHPSCI_THROW("Could not broadcast operands.", 4563);
    return;
}
