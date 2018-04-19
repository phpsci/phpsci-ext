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

#include "initializers.h"
#include "../phpsci.h"
#include <math.h>


/**
 * Create 2D Identity CArray with shape (m,m)
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void identity(CArray * carray, int xy) {
    int i, j;
    #pragma omp parallel for
    for(i = 0; i < xy; i++) {
        for(j = 0; j < xy; j++) {
            carray->array2d[(j * xy) + i] = j == i ? 1.0 : 0.0;
        }
    }
}

/**
 * Create 2D Eye CArray (full of zeros with ones in diagonal with
 * index k).
 *
 * @param rtn_ptr
 * @param x
 * @param y
 * @param k
 */
void eye(MemoryPointer * rtn_ptr, int x, int y, int k)
{
    int i, j;
    carray_init(x, y, rtn_ptr);
    CArray carray = ptr_to_carray(rtn_ptr);
    #pragma omp parallel for
    for(i = 0; i < x; i++) {
        for(j = 0; j < y; j++) {
            carray.array2d[(j * x) + i] = j == (i + k) ? 1.0 : 0.0;
        }
    }
}




/**
 * Create CArray full of zeros
 *
 * zeros    select best function based on shape
 * zeros1d  for 1D CArray
 * zeros2d  for 2D Carray
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void zeros(CArray * carray, int x, int y) {
    if(x > 0 && y > 0) {
        zeros2d(carray, x, y);
    }
    if(x > 0 && y == 0) {
        zeros1d(carray, x);
    }
}
void zeros1d(CArray * carray, int x) {
    int i;
    for(i = 0; i < x; i++) {
        carray->array1d[i] = 0.0;
    }
}
void zeros2d(CArray * carray, int x, int y) {
    int i, j;
    #pragma omp parallel for
    for(i = 0; i < x; i++) {
        for(j = 0; j < y; j++) {
            carray->array2d[(j * x) + i] = 0.0;
        }
    }
}