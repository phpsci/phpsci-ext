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
#include "../phpsci.h"
#include "shape.h"
#include "php.h"

/**
 * Convert Shape config to PHP Array
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void shape_config_to_array(Shape shape, zval * rtn_arr)
{
    int iterator_a;
    array_init(rtn_arr);
    for( iterator_a = 0; iterator_a < shape.dim; ++iterator_a ) {
        add_next_index_long(rtn_arr, (long)shape.config[iterator_a]);
    }
}
