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

#ifndef PHPSCI_EXT_ARITHMETIC_H
#define PHPSCI_EXT_ARITHMETIC_H

#include "../kernel/memory_manager.h"

void add(MemoryPointer * ptr_a, int x_a, int y_a, MemoryPointer * ptr_b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y);
void add_carray_0d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y);
void add_carray_1d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y);
void add_carray_2d(CArray * a, int x_a, int y_a, CArray * b, int x_b, int y_b, MemoryPointer * rtn_ptr, int * size_x, int * size_y);

#endif //PHPSCI_EXT_ARITHMETIC_H
