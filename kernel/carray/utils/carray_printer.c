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

#include "carray_printer.h"

/**
 * Print CArray 2D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param target
 */
void print2d(CArray target, int x, int y) {
    int i, j;
    php_printf("[\n");
    for(i = 0; i < x; i++) {
        php_printf("  [");
        for(j = 0; j < y; j++) {
            php_printf(" %f ", target.array2d[(j * x) + i]);
        }
        php_printf("]\n");
    }
    php_printf("]");
}

/**
 * Print CArray 1D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param target
 */
void print1d(CArray target, int x) {
    int i;
    php_printf("[");
    for(i = 0; i < x; i ++) {
        php_printf(" %f ", target.array1d[i]);
    }
    php_printf("]");
}

/**
 * Print CArray 0D
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param target
 */
void print0d(CArray target) {
    php_printf("%f", target.array0d[0]);
}


/**
 * Print target CArray
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x  Number of rows (2D) or width (1D), if 0, is 0D
 * @param y  Number of cols for 2D matrix
 */
void print_carray(MemoryPointer * ptr, int x, int y) {
    CArray target_carray = ptr_to_carray(ptr);
    if(target_carray.array0d != NULL && x == 0 && y == 0) {
        print0d(target_carray);
        return;
    }
    if(target_carray.array1d != NULL && x > 0 && y == 0) {
        print1d(target_carray, x);
        return;
    }
    if(target_carray.array2d != NULL && x > 0 && y > 0) {
        print2d(target_carray, x , y);
        return;
    }
}


