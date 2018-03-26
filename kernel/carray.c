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

#include "carray.h"
#include "../phpsci.h"
#include "memory_manager.h"
#include "php.h"

/**
 * Initialize CArray space with (rows, cols), if cols = 0, them CArray is treated
 * as array1d.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows  Number of rows
 * @param cols  Number of columns
 */
void carray_init(int rows, int cols, MemoryPointer * ptr) {
    CArray x;
    int j, i;
    x.array2d = (float**)malloc(rows * sizeof(float*) + 64);
    for (i = 0; i < rows; ++i)
        x.array2d[i] = (float*)malloc(cols * sizeof(float));
    add_to_stack(ptr, x,(rows * cols * sizeof(float)) + 64);
}


/**
 * Initialize CArray 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void carray_init1d(int width, MemoryPointer * ptr) {
    CArray x;
    int j, i;
    x.array1d = (float*)malloc(width * sizeof(float) + 64);
    add_to_stack(ptr, x,(width * sizeof(float)) + 64);
}

/**
 * Initialize CArray 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param rows Width
 * @param ptr  MemoryPointer of new CArray
 */
void carray_init0d(MemoryPointer * ptr) {
    CArray x;
    int j, i;
    x.array0d = (float*)malloc(sizeof(float) + 64);
    add_to_stack(ptr, x,sizeof(float) + 64);
}



/**
 *  Get CArray from MemoryPointer
 *
 *  @param ptr      MemoryPointer with target CArray
 *  @return CArray  target CArray
 */
CArray ptr_to_carray(MemoryPointer * ptr) {
    return PHPSCI_MAIN_MEM_STACK.buffer[ptr->uuid];
}

/**
 * Destroy target CArray and set last_deleted_uuid for posterior
 * allocation.
 *
 * @param uuid
 * @param rows
 * @param cols
 */
void destroy_carray(int uuid, int rows, int cols) {
    free(PHPSCI_MAIN_MEM_STACK.buffer[uuid].array2d[0]);
    free(PHPSCI_MAIN_MEM_STACK.buffer[uuid].array2d);
    PHPSCI_MAIN_MEM_STACK.size--;
    PHPSCI_MAIN_MEM_STACK.last_deleted_uuid = uuid;
}

/**
 *  Create MemoryPointer from ZVAL
 *
 *  @param arr zval *           PHP Array to convert
 *  @param pt MemoryPointer *   MemoryPointer
 */
void array_to_carray_ptr(MemoryPointer * ptr, zval * array, int * rows, int * cols) {
    zval * row, * col;
    CArray temp;
    int i =0, j=0;
    *rows = zend_hash_num_elements(Z_ARRVAL_P(array));
    *cols = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), row) {
        ZVAL_DEREF(row);
        if (Z_TYPE_P(row) == IS_ARRAY) {
            if (ptr->uuid == NULL) {
                carray_init(*rows, *cols, ptr);
                temp = ptr_to_carray(ptr);
            }
            convert_to_array(row);
            *cols = zend_hash_num_elements(Z_ARRVAL_P(row));
        } else  {
            if (ptr->uuid == NULL) {
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
                    temp.array2d[i][j] = (float)Z_DVAL_P(col);
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
 * @param carray    CArray to convert
 * @param rtn_array Target ZVAL object
 */
void carray_to_array(CArray carray, zval * rtn_array, int m, int n) {
    int rows, cols;
    zval inner;
    array_init(rtn_array);
    if(n > 0) {
        for( rows = 0; rows < m; rows++ ) {
            array_init(&inner);
            for( cols = 0; cols < n; cols++ ) {
                add_next_index_double(&inner, (float)carray.array2d[rows][cols]);
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
