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

#include "linalg.h"
#include "../phpsci.h"
#include "transformations.h"
#include "../kernel/carray.h"
#include "cblas.h"



void matmul(MemoryPointer * ptr, int n_a_rows, int n_a_cols, MemoryPointer * a_ptr, int n_b_cols, MemoryPointer *b_ptr) {
    int i, j, n_b_rows = n_a_cols;
    MemoryPointer bT_ptr;
    carray_init(n_a_rows, n_b_cols, ptr);
    transpose(&bT_ptr, b_ptr, n_b_rows, n_b_cols);
    CArray bT = ptr_to_carray(&bT_ptr);
    CArray a = ptr_to_carray(a_ptr);
    CArray b = ptr_to_carray(b_ptr);
    CArray rtn = ptr_to_carray(ptr);
    for (i = 0; i < n_a_rows; ++i)
        for (j = 0; j < n_b_cols; ++j)
            rtn.array2d[i][j] = cblas_ddot(n_a_cols, a.array2d[i], 1, bT.array2d[j], 1);
    //mat_destroy(bT);
}