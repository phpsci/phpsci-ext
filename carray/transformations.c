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

#include "transformations.h"
#include "../phpsci.h"
#include "../kernel/carray.h"

/**
 * Transpose a CArray 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void transpose(MemoryPointer * new_ptr, MemoryPointer * target_ptr, int rows, int cols) {
    int i, j;
    carray_init(cols, rows, new_ptr);
    CArray new_arr = ptr_to_carray(new_ptr);
    CArray target_arr = ptr_to_carray(target_ptr);
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            new_arr.array2d[(i * rows) + j] = target_arr.array2d[(j * rows) + i];
        }
    }
}