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

#ifndef PHPSCI_EXT_CARRAY_H
#define PHPSCI_EXT_CARRAY_H
#include "../phpsci.h"



/**
 * PHPSci internal array structure
 *
 * Currently working with shaped 2D, 1D and 0D.
 */
typedef struct CArray {
    float *   array2d;
    float *   array1d;
    float *   array0d;
} CArray;

/**
 * The only thing between PHP and the extension
 */
typedef struct MemoryPointer {
    int uuid;
    int x;
    int y;
} MemoryPointer;

int GET_DIM(int x, int y);
int IS_0D(int x, int y);
int IS_1D(int x, int y);
int IS_2D(int x, int y);

void OBJ_TO_PTR(zval * obj, MemoryPointer * ptr);
void carray_init(int rows, int cols, MemoryPointer * ptr);
void carray_init1d(int width, MemoryPointer * ptr);
void carray_init0d(MemoryPointer * ptr);
void destroy_carray(MemoryPointer * target_ptr);

void array_to_carray_ptr(MemoryPointer * ptr, zval * inarray, int * rows, int * cols);
CArray ptr_to_carray(MemoryPointer * ptr);

void carray_to_array(CArray carray, zval * rtn_array, int m, int n);

void double_to_carray(double input, MemoryPointer * rtn_ptr);

#endif //PHPSCI_EXT_CARRAY_H
