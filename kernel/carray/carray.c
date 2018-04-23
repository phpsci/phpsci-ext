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
#include "../../phpsci.h"
#include "../buffer/memory_manager.h"
#include "../memory_pointer/utils.h"
#include "php.h"

/**
 * Initialize CArray space with (rows, cols), if cols = 0, them CArray is treated
 * as array1d.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows  Number of rows
 * @param cols  Number of columns
 */
void
carray_init(int rows, int cols, MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array2d = (double*)emalloc(rows * cols * sizeof(double));
    x.array1d = NULL;
    x.array0d = NULL;
    ptr->x = rows;
    ptr->y = cols;
    add_to_stack(ptr, x,(rows * cols * sizeof(double)));
}


/**
 * Initialize CArray 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void
carray_init1d(int width, MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array0d = NULL;
    x.array2d = NULL;
    x.array1d = (double*)emalloc(width * sizeof(double));
    ptr->x = width;
    ptr->y = 0;
    add_to_stack(ptr, x,(width * sizeof(double)));
}

/**
 * Initialize CArray 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void
carray_init0d(MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array1d = NULL;
    x.array2d = NULL;
    x.array0d = (double*)emalloc(sizeof(double));
    ptr->x = 0;
    ptr->y = 0;
    add_to_stack(ptr, x,sizeof(double));
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void
OBJ_TO_PTR(zval * obj, MemoryPointer * ptr)
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
CArray
ptr_to_carray(MemoryPointer * ptr)
{
    return PHPSCI_MAIN_MEM_STACK.buffer[ptr->uuid];
}

/**
 *  Get CArray Pointer from MemoryPointer
 *
 *  @author Henrique Borba <henrique.borba.dev@gmail.com>
 *  @param ptr      MemoryPointer with target CArray
 *  @return CArray  target CArray
 */
CArray *
ptr_to_carray_ptr(MemoryPointer * ptr)
{
    return &(PHPSCI_MAIN_MEM_STACK.buffer[ptr->uuid]);
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
void
destroy_carray(MemoryPointer * target_ptr)
{
    if(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array2d != NULL) {
        efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array2d);
        return;
    }
    if(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array1d != NULL) {
        efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array1d);
        return;
    }
    if(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array0d != NULL) {
        efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array0d);
        return;
    }
    PHPSCI_MAIN_MEM_STACK.last_deleted_uuid = target_ptr->uuid;
}

/**
 * Create ZVAL_ARR from CArray
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param carray    CArray to convert
 * @param rtn_array Target ZVAL object
 */
void
carray_to_array(CArray carray, zval * rtn_array, int m, int n)
{
    int rows, cols;
    zval inner;
    array_init(rtn_array);
    if(n > 0) {
        for( rows = 0; rows < m; rows++ ) {
            array_init(&inner);
            for( cols = 0; cols < n; cols++ ) {
                add_next_index_double(&inner, carray.array2d[(cols * m) + rows]);
            }
            add_next_index_zval(rtn_array, &inner);
        }
    }
    if(n == 0) {
        for( rows = 0; rows < m; rows++ ) {
            add_next_index_double(rtn_array, carray.array1d[rows]);
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
void
double_to_carray(double input, MemoryPointer * rtn_ptr)
{
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = input;
}

/**
 * Set CArray Value
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param index
 * @param value
 */
void
carray_set_value(MemoryPointer * ptr_a, Tuple * index, double value)
{
    CArray target_carray = ptr_to_carray(ptr_a);
    if(IS_2D(ptr_a)) {
        target_carray.array2d[(index->t[1] * ptr_a->x) + index->t[0]] = value;
        return;
    }
    if(IS_1D(ptr_a)) {
        target_carray.array1d[index->t[0]] = value;
        return;
    }
    if(IS_0D(ptr_a)) {
        target_carray.array0d[0] = value;
        return;
    }
}

/**
 * Get CArray Value
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param index
 */
double
carray_get_value(MemoryPointer * ptr_a, Tuple * index)
{
    CArray target_carray = ptr_to_carray(ptr_a);
    if(IS_2D(ptr_a)) {
        return target_carray.array2d[(index->t[1] * ptr_a->x) + index->t[0]];
    }
    if(IS_1D(ptr_a)) {
        return target_carray.array1d[index->t[0]];
    }
    if(IS_0D(ptr_a)) {
        return target_carray.array0d[0];
    }
}