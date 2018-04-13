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

#ifndef PHPSCI_EXT_BASIC_OPERATIONS_H
#define PHPSCI_EXT_BASIC_OPERATIONS_H
#include "../kernel/memory_manager.h"

void sum_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y);
void sum_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis, int * size_x, int * size_y);

        void sub_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y);
void sub_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis);

#endif //PHPSCI_EXT_BASIC_OPERATIONS_H
