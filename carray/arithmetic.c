/*
  +----------------------------------------------------------------------+
  | PHP Version 7 | PHPSci                                               |
  +----------------------------------------------------------------------+
  | Copyright (c) 2018 Henrique Borba                                    |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: Henrique Borba <henrique.borba.dev@gmail.com>                |
  +----------------------------------------------------------------------+
*/

#include "arithmetic.h"
#include "../phpsci.h"
#include "../kernel/carray.h"
#include "initializers.h"
#include "zend_exceptions.h"

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
                rtn_arr.array2d[i][j] = a->array2d[i][j] + b->array0d[0];
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
                rtn_arr.array2d[i][j] = a->array2d[i][j] + b->array1d[j];
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
                rtn_arr.array2d[i][j] = a->array2d[i][j] + b->array2d[i][j];
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
                rtn_arr.array2d[i][j] = a->array2d[i][j] + b->array2d[0][j];
            }
        }
        return;
    }
    PHPSCI_THROW("Could not broadcast operands.", 4563);
    return;
}
