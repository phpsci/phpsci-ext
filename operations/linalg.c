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
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"
#include "cblas.h"
#include "lapacke.h"

/**
 * Program computes the singular value decomposition of a general rectangular matrix A
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param target_ptr
 */
void
svd(MemoryPointer * a_ptr, MemoryPointer * rtn_ptr, MemoryPointer * singularvalues_ptr,
    MemoryPointer * left_vectors_ptr, MemoryPointer * right_vectors_ptr)
{
    double u[a_ptr->x*a_ptr->x], vt[a_ptr->y*a_ptr->y];

    // Initialize Singular Vectors array
    carray_init1d(a_ptr->y, singularvalues_ptr);
    CArray singular_values_arr = ptr_to_carray(singularvalues_ptr);
    // Initialize singular Left Vectors Array
    carray_init(a_ptr->x, a_ptr->x, left_vectors_ptr);
    CArray left_vectors_arr = ptr_to_carray(left_vectors_ptr);
    // Initialize singular Right vectors Array
    carray_init(a_ptr->y, a_ptr->y, right_vectors_ptr);
    CArray right_vectors_arr = ptr_to_carray(right_vectors_ptr);

    lapack_int m = (lapack_int)a_ptr->x, n = (lapack_int)a_ptr->y;
    lapack_int lda = (lapack_int)m, ldu = (lapack_int)m, ldvt = (lapack_int)n, info;
    // Compute SVD
    CArray target_carray = ptr_to_carray(a_ptr);
    carray_init(a_ptr->x, a_ptr->y, rtn_ptr);
    CArray rtn_carray = ptr_to_carray(rtn_ptr);
    memcpy(rtn_carray.array2d, target_carray.array2d, (a_ptr->x * a_ptr->y * sizeof(double)));
    info = LAPACKE_dgesdd( LAPACK_COL_MAJOR, 'S', m, n, rtn_carray.array2d, lda, singular_values_arr.array1d,
                           left_vectors_arr.array2d, ldu, right_vectors_arr.array2d, ldvt );
}


/**
 * Compute the (multiplicative) inverse of a matrix.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param m
 * @param target_ptr
 * @param rtn_ptr
 */
void
inv(MemoryPointer * target_ptr, MemoryPointer * rtn_ptr)
{
    lapack_int * ipiv = safe_emalloc(target_ptr->x, sizeof(lapack_int), 0);
    lapack_int ret, m, n, lda;
    // Load CArrays
    CArray target_carray = ptr_to_carray(target_ptr);
    carray_init(target_ptr->x, target_ptr->y, rtn_ptr);
    CArray rtn_carray = ptr_to_carray(rtn_ptr);
    memcpy(rtn_carray.array2d, target_carray.array2d, (target_ptr->x * target_ptr->y * sizeof(double)));
    // Use LAPACKE to calculate
    m = target_ptr->x;
    n = target_ptr->y;
    lda = target_ptr->x;
    ret =  LAPACKE_dgetrf(LAPACK_COL_MAJOR,
                          (lapack_int) m,
                          (lapack_int) n,
                          rtn_carray.array2d,
                          (lapack_int) lda,
                          ipiv);

    ret = LAPACKE_dgetri(LAPACK_COL_MAJOR,
                         (lapack_int) n,
                         rtn_carray.array2d,
                         (lapack_int) lda,
                         ipiv);
}

/**
 * Outer product of matrices
 *
 * @param ptr_a
 * @param ptr_b
 * @param ptr_rtn
 */
void
outer(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * ptr_rtn)
{
    // Get CArray A and B from buffer
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    // Initialize result matrix
    carray_init((ptr_a->x * ptr_a->y), (ptr_a->x * ptr_a->y), ptr_rtn);
    CArray carray_rtn = ptr_to_carray(ptr_rtn);
    cblas_dger(CblasColMajor, /* you’re using row-major storage */
               ptr_a->x,           /* the matrix X has dx1 rows ...  */
               ptr_b->y,           /*  ... and dx2 columns.          */
               1.0,           /* scale factor to apply to x1x2' */
               carray_a.array2d,
               1,             /* stride between elements of x1. */
               carray_b.array2d,
               1,             /* stride between elements of x2. */
               carray_rtn.array2d,
               ptr_b->y);
}

/**
 * Inner product of matrices
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param a
 * @param b
 */
void
inner(int * rtn_x, int * rtn_y, MemoryPointer * ptr, int x_a, int y_a, MemoryPointer * a_ptr, int x_b, int y_b, MemoryPointer * b_ptr)
{
    int i, j, k;
    CArray a = ptr_to_carray(a_ptr);
    CArray b = ptr_to_carray(b_ptr);

    if (IS_1D(a_ptr) && IS_1D(b_ptr)) {
        carray_init0d(ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        rtn_arr.array0d[0] = 0;
        for(i = 0; i < x_a; i++) {
            rtn_arr.array0d[0] += a.array1d[i] * b.array1d[i];
        }
        *rtn_x = 0;
        return;
    }
    if (IS_1D(a_ptr) && IS_0D(b_ptr)) {
        carray_init1d(x_a,  ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            rtn_arr.array1d[i] = a.array1d[i] * b.array0d[0];
        }
        *rtn_x = x_a;
        return;
    }
    if (IS_2D(a_ptr) && IS_2D(b_ptr)) {
        carray_init(x_a, x_a, ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            for(j = 0; j < x_a; j++) {
                rtn_arr.array2d[(j * x_a) + i] = 0;
                for(k = 0; k < y_b; k++) {
                    rtn_arr.array2d[(j * x_a) + i] += a.array2d[(k * x_a) + i] * b.array2d[(k * x_a) + j];
                }
            }
        }
        *rtn_x = x_a;
        *rtn_y = x_a;
        return;
    }
    if (IS_2D(a_ptr) && IS_0D(b_ptr)) {
        carray_init(x_a, y_a, ptr);
        CArray rtn_arr = ptr_to_carray(ptr);
        for(i = 0; i < x_a; i++) {
            for(j = 0; j < y_a; j++) {
                rtn_arr.array2d[(j * x_a) + i] = a.array2d[(j * x_a) + i] * b.array0d[0];
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
void
matmul(MemoryPointer * ptr, int n_a_rows, int n_a_cols, MemoryPointer * a_ptr, int n_b_cols, MemoryPointer * b_ptr) {
    int i, j, n_b_rows = n_a_cols;
    MemoryPointer bT_ptr;
    CArray a = ptr_to_carray(a_ptr);
    CArray b = ptr_to_carray(b_ptr);
    if(n_b_cols > 0 && n_a_cols > 0) {
        carray_init(n_a_rows, n_b_cols, ptr);
        CArray rtn = ptr_to_carray(ptr);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n_a_rows, n_b_cols, n_a_cols, 1.0f, a.array2d, n_a_rows, b.array2d, n_b_rows, 0.0f, rtn.array2d, n_a_rows);
        return;
    }
    if(n_b_cols == 0 && n_a_cols > 0) {
        carray_init1d(n_a_rows, ptr);
        CArray rtn = ptr_to_carray(ptr);
        transpose(&bT_ptr, a_ptr);
        CArray bT = ptr_to_carray(&bT_ptr);
        for (i = 0; i < n_a_rows; ++i) {
            rtn.array1d[i] = cblas_ddot(n_a_rows, bT.array2d, 1, b.array1d, 1);
        }
        return;
    }
    if(n_b_cols > 0 && n_a_cols == 0) {
        carray_init1d(n_a_rows, ptr);
        CArray rtn = ptr_to_carray(ptr);
        transpose(&bT_ptr, b_ptr);
        CArray bT = ptr_to_carray(&bT_ptr);
        for (i = 0; i < n_a_rows; ++i) {
            rtn.array1d[i] = cblas_ddot(n_a_rows, bT.array2d, 1, a.array1d, 1);
        }
        return;
    }
    if(n_b_cols == 0 && n_a_cols == 0) {
        carray_init0d(ptr);
        CArray rtn = ptr_to_carray(ptr);
        rtn.array0d[0] = cblas_ddot(n_a_rows, a.array1d, 1, b.array1d, 1);
        return;
    }
}
