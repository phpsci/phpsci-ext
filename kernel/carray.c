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
#include "exceptions.h"
#include "php.h"

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 *
 * @return
 */
int PTR_TO_DIM(MemoryPointer * ptr)
{
    return GET_DIM(ptr->x, ptr->y);
}

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
void carray_init(Shape init_shape, MemoryPointer * ptr)
{
    //@todo Implement universal carray_init with new CArray architecture.
}

/**
 * Initialize CArray 2D
 *
 * @deprecated This will be deprecated in favor of new CArray architecture.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows  Number of rows
 * @param cols  Number of columns
 */
void carray_init2d(int rows, int cols, MemoryPointer * ptr)
{
    CArray x;
    int j, i;
    x.array2d = (double*)emalloc(rows * cols * sizeof(double));
    x.array1d = NULL;
    x.array0d = NULL;
    add_to_stack(ptr, x,(rows * cols * sizeof(double)));
}


/**
 * Initialize CArray 1D
 *
 * @deprecated This will be deprecated in favor of new CArray architecture.
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
    x.array1d = (double*)emalloc(width * sizeof(double));
    add_to_stack(ptr, x,(width * sizeof(double)));
}

/**
 * Initialize CArray 0D
 *
 * @deprecated This will be deprecated in favor of new CArray architecture.
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
    x.array0d = (double*)emalloc(sizeof(double));
    add_to_stack(ptr, x,sizeof(double));
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
    if(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array != NULL) {
        efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array);
        return;
    }
    if(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array_shape != NULL) {
        efree(PHPSCI_MAIN_MEM_STACK.buffer[target_ptr->uuid].array_shape);
        return;
    }
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
void carray_to_array(CArray carray, zval * rtn_array, int m, int n)
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
        // If 2-D, fill inside values.
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
void double_to_carray(double input, MemoryPointer * rtn_ptr)
{
    carray_init0d(rtn_ptr);
    CArray rtn_arr = ptr_to_carray(rtn_ptr);
    rtn_arr.array0d[0] = input;
}

/**
 * Operations are valid when they matches one of the following rules:
 *
 * - Both matrices has the same shape
 * - One of the matrices are 1-D
 *
 * Return 0 if invalid, 1 if valid and 2 if matrices are valid but need to
 * be switched.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
int validate_carray_arithmetic_broadcast(CArray a, CArray b, int * broadcasted_shape)
{
    return 1;
}

/**
 * Broadcast N-Dimensional CArray to user defined operation.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void carray_broadcast_arithmetic(MemoryPointer * a, MemoryPointer * b, MemoryPointer * rtn_ptr,
                      void cFunction(MemoryPointer * , MemoryPointer * , MemoryPointer *)
)
{
    CArray arr_a = ptr_to_carray(a);
    CArray arr_b = ptr_to_carray(b);
    int validate_rtn, * broadcasted_shape;
    int a_x, b_x, a_y, b_y;
    // Check if operation is valid.
    validate_rtn = validate_carray_arithmetic_broadcast(arr_a, arr_b, broadcasted_shape);
    if( validate_rtn == 0 ) {
        // Invalid Shapes
        throw_could_not_broadcast_exception("Could not broadcast provided matrices.");
        return;
    }
    if( validate_rtn == 1 ) {
        // Call arithmetic function with provided operands order. (eg: a + b)
        cFunction(a, b, rtn_ptr);
        return;
    }
    if( validate_rtn == 2 ) {
        // Call arithmetic function with switched operands. (eg: b + a)
        cFunction(b, a, rtn_ptr);
        return;
    }
}