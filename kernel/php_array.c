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

#include "php_array.h"
#include "../phpsci.h"
#include "php.h"


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
    CArray * temp;
    int i =0, j=0, dim = 0;
    int iterator;
    *rows = zend_hash_num_elements(Z_ARRVAL_P(array));
    *cols = 0;
    array_dim(array, &dim);
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), row) {
        ZVAL_DEREF(row);
        if (Z_TYPE_P(row) == IS_ARRAY) {
            *cols = zend_hash_num_elements(Z_ARRVAL_P(row));
            if (ptr->uuid == UNINITIALIZED) {
                carray_init2d(*rows, *cols, ptr);
                temp = ptr_to_carray_ref(ptr);
            }
            convert_to_array(row);

        } else  {
            if (ptr->uuid == UNINITIALIZED) {
                carray_init1d(*rows, ptr);
                temp = ptr_to_carray_ref(ptr);
            }
        }
    } ZEND_HASH_FOREACH_END();
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(array), row) {
        ZVAL_DEREF(row);
        if (Z_TYPE_P(row) == IS_ARRAY) {
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(row), col) {
                convert_to_double(col);
                temp->array2d[(j * *rows) + i] = (double)Z_DVAL_P(col);
                ++j;
            } ZEND_HASH_FOREACH_END();
        } else {
            convert_to_double(row);
            temp->array1d[i] = (double)Z_DVAL_P(row);
        }
        j = 0;
        ++i;
    } ZEND_HASH_FOREACH_END();
    temp->array_shape.dim = dim;
    temp->array_shape.config = NULL;
    array_shape(array, &(temp->array_shape), dim, &iterator);
}

/**
 * Get PHP Array number of dimensions
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param  zval*          Target Array
 * @param  int*
 */
void array_dim(zval * array, int * dim)
{
    zval *entry = NULL;
    if (Z_TYPE_P(array) == IS_ARRAY) {
        *dim = *dim + 1;
        entry = zend_hash_get_current_data(Z_ARRVAL_P(array));
        array_dim(entry, dim);
    }
    return;
}

/**
 * Get PHP array shape
 *
 * @param arr
 * @param shape
 */
void array_shape(zval * arr, Shape * shape, int dim, int * iterator)
{
    zval *entry = NULL;
    // Initialize Shape
    if(shape->config == NULL) {
        shape->config = (int *) emalloc(dim * sizeof(int));
        *iterator = 0;
    }
    // Iterate over dimensions
    if (Z_TYPE_P(arr) == IS_ARRAY) {
        shape->config[*iterator] = (int)zend_hash_num_elements(Z_ARRVAL_P(arr));
        entry = zend_hash_get_current_data(Z_ARRVAL_P(arr));
        *iterator = *iterator + 1;
        array_shape(entry, shape, dim, iterator);
    }
    return;
}
