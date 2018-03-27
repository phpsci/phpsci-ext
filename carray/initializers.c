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
            carray->array2d[i][j] = j == i ? 1.0 : 0.0;
        }
    }
}



/**
 * Return numbers spaced evenly on a log scale.
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void logspace(MemoryPointer * ptr, float start, float stop, int num, float base) {
    int i;
    float step = (stop - start)/(num - 1);
    carray_init1d(num, ptr);
    CArray new_array = ptr_to_carray(ptr);
    for(i = 0; i < num; ++i) {
        new_array.array1d[i] = pow(base, (start + (i*step)));
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
            carray->array2d[i][j] = 0.0;
        }
    }
}