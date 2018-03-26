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
#include "../phpsci.h"
#include "../kernel/carray.h"

#ifndef PHPSCI_EXT_RANGES_H
#define PHPSCI_EXT_RANGES_H
#include "../kernel/memory_manager.h"

void arange(MemoryPointer * new_ptr, float start, float stop, float step, int * width);
void linspace(MemoryPointer * ptr, float start, float stop, int num);
#endif //PHPSCI_EXT_RANGES_H
