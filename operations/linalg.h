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

#ifndef PHPSCI_EXT_LINALG_H
#define PHPSCI_EXT_LINALG_H


#include "../kernel/buffer/memory_manager.h"
void svd(MemoryPointer * a_ptr, MemoryPointer * rtn_ptr, MemoryPointer * singularvalues_ptr, MemoryPointer * left_vectors_ptr, MemoryPointer * right_vectors_ptr);
void inv(MemoryPointer * target_ptr, MemoryPointer * rtn_ptr);
void inner(int * rtn_x, int * rtn_y, MemoryPointer * ptr, int x_a, int y_a, MemoryPointer * a_ptr, int x_b, int y_b, MemoryPointer * b_ptr);
void outer(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * ptr_rtn);
void matmul(MemoryPointer * ptr, int n_a_rows, int n_a_cols, MemoryPointer * a_ptr, int n_b_cols, MemoryPointer *b_ptr);
#endif //PHPSCI_EXT_LINALG_H
