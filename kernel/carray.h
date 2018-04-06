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

#ifndef PHPSCI_EXT_CARRAY_H
#define PHPSCI_EXT_CARRAY_H
#include "../phpsci.h"



/**
 * PHPSci internal array structure
 *
 * Currently working with shaped 2D, 1D and 0D.
 */
typedef struct CArray {
    float **  array2d;
    float *   array1d;
    float *   array0d;
} CArray;

/**
 * The only thing between PHP and the extension
 */
typedef struct MemoryPointer {
    int uuid;
} MemoryPointer;

int GET_DIM(int x, int y);
int IS_0D(int x, int y);
int IS_1D(int x, int y);
int IS_2D(int x, int y);

void carray_init(int rows, int cols, MemoryPointer * ptr);
void carray_init1d(int width, MemoryPointer * ptr);
void carray_init0d(MemoryPointer * ptr);
void destroy_carray(int uuid, int rows, int cols);

void array_to_carray_ptr(MemoryPointer * ptr, zval * inarray, int * rows, int * cols);
CArray ptr_to_carray(MemoryPointer * ptr);

void carray_to_array(CArray carray, zval * rtn_array, int m, int n);

#endif //PHPSCI_EXT_CARRAY_H
