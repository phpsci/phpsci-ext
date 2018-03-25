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
            new_arr.array2d[j][i] = target_arr.array2d[i][j];
        }
    }
}