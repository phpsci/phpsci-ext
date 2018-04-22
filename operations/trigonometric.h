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

#ifndef PHPSCI_EXT_TRIGONOMETRIC_H
#define PHPSCI_EXT_TRIGONOMETRIC_H

#include "../kernel/buffer/memory_manager.h"

void sin_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);
void cos_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);
void tan_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);

void arcsin_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);
void arccos_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);
void arctan_carray(MemoryPointer * ptr, MemoryPointer * rtn_ptr);
#endif //PHPSCI_EXT_TRIGONOMETRIC_H
