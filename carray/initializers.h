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

#ifndef PHPSCI_EXT_INITIALIZERS_H
#define PHPSCI_EXT_INITIALIZERS_H
#include "../phpsci.h"
#include "../kernel/carray.h"
void identity(CArray * carray, int xy);
void logspace(MemoryPointer * ptr, float start, float stop, int num, float base);

void zeros2d(CArray * carray, int x, int y);
void zeros(CArray * carray, int x, int y);
void zeros1d(CArray * carray, int x);
#endif //PHPSCI_EXT_INITIALIZERS_H
