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
#include "random.h"
#include "../phpsci.h"
#include "../kernel/carray/carray.h"
#include "lapacke.h"
#include <math.h>
#include <stdlib.h>


/**
 * CArray samples from a standard Normal distribution
 *
 * @author Henrique Borba
 * @param ptr
 */
void standard_normal(MemoryPointer * ptr, int seed, int x, int y)
{
    int i, j, info;
    lapack_int seed_vector[4];
    if(x > 0 && y == 0) {
        carray_init1d(x, ptr);
        CArray new_array = ptr_to_carray(ptr);
        info = LAPACKE_dlarnv(3, seed_vector, x, new_array.array1d);
    }
}