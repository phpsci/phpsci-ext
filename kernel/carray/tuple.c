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
#include "tuple.h"
#include "php.h"

/**
 * ZVAL to Tuple
 *
 * @param obj
 * @param tuple
 */
void OBJ_TO_TUPLE(zval * obj, Tuple * tuple)
{
    int iterator_count = 0;
    zval * temp_val;
    tuple->size = (int)zend_hash_num_elements(Z_ARRVAL_P(obj));
    init_tuple(tuple->size, tuple);
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(obj), temp_val)
    {
        convert_to_double(temp_val);
        tuple->t[iterator_count] = (double)Z_DVAL_P(temp_val);
        ++iterator_count;
    } ZEND_HASH_FOREACH_END();
}

/**
 * Initialize Tuple
 *
 * @param size
 * @param tuple
 */
void
init_tuple(int size, Tuple * tuple)
{
    tuple->t = emalloc(size * sizeof(int));
    tuple->size = size;
}

/**
 * Free Tuple structure from buffer
 *
 * @param tuple
 */
void
free_tuple(Tuple * tuple)
{
    efree(tuple->t);
}
