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

#ifndef PHPSCI_EXT_BASIC_OPERATIONS_H
#define PHPSCI_EXT_BASIC_OPERATIONS_H
#include "../kernel/memory_manager.h"

void sum_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y);
void sum_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis);

void sub_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y);
void sub_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis);

#endif //PHPSCI_EXT_BASIC_OPERATIONS_H
