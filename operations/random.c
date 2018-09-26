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
#include <time.h>


/**
 * CArray samples from a standard Normal distribution
 *
 * @author Henrique Borba
 * @param ptr
 */
void
standard_normal(MemoryPointer * ptr, int seed, int x, int y)
{
    int i, j, info;
    lapack_int seed_vector[4];
    if(x > 0 && y == 0) {
        carray_init1d(x, ptr);
        CArray new_array = ptr_to_carray(ptr);
        info = LAPACKE_dlarnv(2, seed_vector, x, new_array.array1d);
    }
}

/**
 *
 * @param length
 */
void
randint(MemoryPointer * rtn_ptr, int length)
{
    if(!length) {
        carray_init0d(rtn_ptr);
        CArray rtn_carray = ptr_to_carray(rtn_ptr);
        srand(time(0));
        rtn_carray.array0d[0] = rand();
        return;
    }

    carray_init1d(length, rtn_ptr);
    CArray rtn_carray = ptr_to_carray(rtn_ptr);
    for(int i = 0; i < length; i++) {
        srand(time(0));
        rtn_carray.array1d[i] = rand();
    }
    return;
}


/**
 * @param rtn_ptr
 * @see https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
 */
void
randn(MemoryPointer * rtn_ptr, int x, int y)
{

}