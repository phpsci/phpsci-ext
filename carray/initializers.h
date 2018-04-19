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

#ifndef PHPSCI_EXT_INITIALIZERS_H
#define PHPSCI_EXT_INITIALIZERS_H
#include "../phpsci.h"
#include "../kernel/carray.h"
void identity(CArray * carray, int xy);
void eye(MemoryPointer * rtn_ptr, int x, int y, int k);

void zeros2d(CArray * carray, int x, int y);
void zeros(MemoryPointer * ptr, int x, int y);
void zeros1d(CArray * carray, int x);

void ones(MemoryPointer * ptr, int x, int y);
void ones1d(CArray * carray, int x);
void ones2d(CArray * carray, int x, int y);

void full(MemoryPointer * ptr, int x, int y, double num);
void full1d(CArray * carray, int x, double num);
void full2d(CArray * carray, int x, int y, double num);
#endif //PHPSCI_EXT_INITIALIZERS_H
