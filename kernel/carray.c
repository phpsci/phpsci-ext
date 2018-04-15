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

#include "carray.h"
#include "../phpsci.h"
#include "memory_manager.h"
#include "php.h"


/**
 * Get CArray dimensions based on X and Y
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int GET_DIM(int x, int y)
{
    if(x == 0 && y == 0)
        return 0;
    if(x > 0 && y == 0)
        return 1;
    if(x > 0 && y > 0)
        return 2;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int IS_0D(int x, int y)
{
    if(x == 0 && y == 0)
        return 1;
    return 0;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int IS_1D(int x, int y)
{
    if(x > 0 && y == 0)
        return 1;
    return 0;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int IS_2D(int x, int y)
{
    if(x > 0 && y > 0)
        return 1;
    return 0;
}

/**
 * Initialize CArray space with (rows, cols), if cols = 0, them CArray is treated
 * as array1d.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows  Number of rows
 * @param cols  Number of columns
 */
void carray_init(int rows, int cols, MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array2d = (float*)emalloc(rows * cols * sizeof(float));
    x.array1d = NULL;
    x.array0d = NULL;
    add_to_stack(ptr, x,(rows * cols * sizeof(float)));
}


/**
 * Initialize CArray 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void carray_init1d(int width, MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array0d = NULL;
    x.array2d = NULL;
    x.array1d = (float*)emalloc(width * sizeof(float) + 64);
    add_to_stack(ptr, x,(width * sizeof(float)) + 64);
}

/**
 * Initialize CArray 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void carray_init0d(MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array1d = NULL;
    x.array2d = NULL;
    x.array0d = (float*)emalloc(sizeof(float) + 64);
    add_to_stack(ptr, x,sizeof(float) + 64);
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void OBJ_TO_PTR(zval * obj, MemoryPointer * ptr)
{
    zval rv;
    ptr->uuid = (int)zval_get_long(zend_read_property(phpsci_sc_entry, obj, "uuid", sizeof("uuid") - 1, 1, &rv));
    ptr->x = (int)zval_get_long(zend_read_property(phpsci_sc_entry, obj, "x", sizeof("x") - 1, 1, &rv));
    ptr->y = (int)zval_get_long(zend_read_property(phpsci_sc_entry, obj, "y", sizeof("y") - 1, 1, &rv));
}

/**
 *  Get CArray from MemoryPointer
 *
 *  @author Henrique Borba <henrique.borba.dev@gmail.com>
 *  @param ptr      MemoryPointer with target CArray
 *  @return CArray  target CArray
 */
CArray ptr_to_carray(MemoryPointer * ptr)
{
    return PHPSCI_MAIN_MEM_STACK.buffer[ptr->uuid];
}

/**
 * Destroy target CArray and set last_deleted_uuid for posterior
 * allocation.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param uuid  UUID of CArray to be destroyed
 * @param rows  Number of rows in CArray to be destroyed
 * @param cols  Number os cols in CArray to be destroyed
 */
void destroy_carray(MemoryPointer * target_ptr)
{
    efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array2d);
    PHPSCI_MAIN_MEM_STACK.size--;
    PHPSCI_MAIN_MEM_STACK.last_deleted_uuid = target_ptr->uuid;
}

/**
 * Create MemoryPointer from ZVAL
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param arr zval *           PHP Array to convert
 * @param pt MemoryPointer *   MemoryPointer
 */
void array_to_carray_ptr(MemoryPointer * ptr, zval * array, int * rows, int * cols)
{
    zval * row, * col;
    CArray temp;
    int i =0, j=0;
    *rows = zend_hash_num_elements(Z_ARRVAL_P(array));
    *cols = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), row) {
        ZVAL_DEREF(row);
        if (Z_TYPE_P(row) == IS_ARRAY) {
            *cols = zend_hash_num_elements(Z_ARRVAL_P(row));
            if (ptr->uuid == UNINITIALIZED) {
                carray_init(*rows, *cols, ptr);
                temp = ptr_to_carray(ptr);
            }
            convert_to_array(row);

        } else  {
            if (ptr->uuid == UNINITIALIZED) {
                carray_init1d(*rows, ptr);
                temp = ptr_to_carray(ptr);
            }
        }
    } ZEND_HASH_FOREACH_END();
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), row) {
        ZVAL_DEREF(row);
        if (Z_TYPE_P(row) == IS_ARRAY) {
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(row), col) {
                convert_to_double(col);
                temp.array2d[(j * *rows) + i] = (float)Z_DVAL_P(col);
                ++j;
            } ZEND_HASH_FOREACH_END();
        } else {
            convert_to_double(row);
            temp.array1d[i] = (float)Z_DVAL_P(row);
        }
        j = 0;
        ++i;
    } ZEND_HASH_FOREACH_END();
}


/**
 * Create ZVAL_ARR from CArray
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param carray    CArray to convert
 * @param rtn_array Target ZVAL object
 */
void carray_to_array(CArray carray, zval * rtn_array, int m, int n)
{
    int rows, cols;
    zval inner;
    array_init(rtn_array);
    if(n > 0) {
        for( rows = 0; rows < m; rows++ ) {
            array_init(&inner);
            for( cols = 0; cols < n; cols++ ) {
                add_next_index_double(&inner, (float)carray.array2d[(cols * m) + rows]);
            }
            add_next_index_zval(rtn_array, &inner);
        }
    }
    if(n == 0) {
        for( rows = 0; rows < m; rows++ ) {
            add_next_index_double(rtn_array, (float)carray.array1d[rows]);
        }
    }
}

/**
 * Create CArray from Double
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param carray
 * @param rtn_array
 */
void double_to_carray(double input, MemoryPointer * rtn_ptr)
{
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = input;
}
