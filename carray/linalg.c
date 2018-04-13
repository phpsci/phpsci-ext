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

#include "linalg.h"
#include "../phpsci.h"
#include "transformations.h"
#include "../kernel/carray.h"
#include "cblas.h"


/**
 * Inner product of matrices
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param a
 * @param b
 */
void inner(int * rtn_x, int * rtn_y, MemoryPointer * ptr, int x_a, int y_a, MemoryPointer * a_ptr, int x_b, int y_b, MemoryPointer * b_ptr)
{
    int i, j, k;
    CArray a = ptr_to_carray(a_ptr);
    CArray b = ptr_to_carray(b_ptr);

    if (IS_1D(x_a, y_a) && IS_1D(x_b, y_b)) {
        carray_init0d(ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        rtn_arr.array0d[0] = 0;
        for(i = 0; i < x_a; i++) {
            rtn_arr.array0d[0] += a.array1d[i] * b.array1d[i];
        }
        *rtn_x = 0;
        return;
    }
    if (IS_1D(x_a, y_a) && IS_0D(x_b, y_b)) {
        carray_init1d(x_a,  ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            rtn_arr.array1d[i] = a.array1d[i] * b.array0d[0];
        }
        *rtn_x = x_a;
        return;
    }
    if (IS_2D(x_a, y_a) && IS_2D(x_b, y_b)) {
        carray_init(x_a, x_a, ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            for(j = 0; j < x_a; j++) {
                rtn_arr.array2d[i][j] = 0;
                for(k = 0; k < y_b; k++) {
                    rtn_arr.array2d[i][j] += a.array2d[i][k] * b.array2d[j][k];
                }
            }
        }
        *rtn_x = x_a;
        *rtn_y = x_a;
        return;
    }
    if (IS_2D(x_a, y_a) && IS_0D(x_b, y_b)) {
        carray_init(x_a, y_a, ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            for(j = 0; j < y_a; j++) {
                rtn_arr.array2d[i][j] = a.array2d[i][j] * b.array0d[0];
            }
        }
        *rtn_x = x_a;
        *rtn_y = y_a;
        return;
    }
}



/**
 * Matrix product of two arrays.
 *
 * If both CArrays are 2-D they are multiplied like conventional matrices.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr       Destination MemoryPointer
 * @param n_a_rows  Number of rows in CArray A
 * @param n_a_cols  Number of cols in CArray B
 * @param a_ptr     CArray A MemoryPointer
 * @param n_b_cols  Number of cols in CArray B
 * @param b_ptr     CArray B MemoryPointer
 */
void matmul(MemoryPointer * ptr, int n_a_rows, int n_a_cols, MemoryPointer * a_ptr, int n_b_cols, MemoryPointer * b_ptr) {
    int i, j, n_b_rows = n_a_cols;
    MemoryPointer bT_ptr;
    CArray a = ptr_to_carray(a_ptr);
    CArray b = ptr_to_carray(b_ptr);
    if(n_b_cols > 0 && n_a_cols > 0) {
        carray_init(n_a_rows, n_b_cols, ptr);
        transpose(&bT_ptr, b_ptr, n_b_rows, n_b_cols);
        CArray bT = ptr_to_carray(&bT_ptr);
        CArray rtn = ptr_to_carray(ptr);
        for (i = 0; i < n_a_rows; ++i) {
            for (j = 0; j < n_b_cols; ++j) {
                rtn.array2d[i][j] = cblas_sdot(n_a_cols, a.array2d[i], 1, bT.array2d[j], 1);
            }
        }
        return;
    }
    if(n_b_cols == 0 && n_a_cols > 0) {
        carray_init1d(n_a_rows, ptr);
        CArray rtn = ptr_to_carray(ptr);
        for (i = 0; i < n_a_rows; ++i) {
            rtn.array1d[i] = cblas_sdot(n_a_cols, a.array2d[i], 1, b.array1d, 1);
        }

        return;
    }
    if(n_b_cols > 0 && n_a_cols == 0) {
        carray_init1d(n_a_rows, ptr);
        CArray rtn = ptr_to_carray(ptr);
        transpose(&bT_ptr, b_ptr, n_a_rows, n_b_cols);
        CArray bT = ptr_to_carray(&bT_ptr);
        for (i = 0; i < n_a_rows; ++i) {
            rtn.array1d[i] = cblas_sdot(n_a_rows, bT.array2d[i], 1,a.array1d, 1);
        }
        return;
    }
    if(n_b_cols == 0 && n_a_cols == 0) {
        carray_init0d(ptr);
        CArray rtn = ptr_to_carray(ptr);
        rtn.array0d[0] = cblas_sdot(n_a_rows, a.array1d, 1, b.array1d, 1);
        return;
    }
}
