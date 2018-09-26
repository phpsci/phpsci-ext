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

#include "set_routines.h"
#include "../phpsci.h"
#include "initializers.h"
#include "../kernel/carray/carray.h"
#include "../kernel/memory_pointer/utils.h"

/**
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
in1d(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
   CArray array_a = ptr_to_carray(ptr_a);
   CArray array_b = ptr_to_carray(ptr_b);
   zeros(rtn_ptr, ptr_a->x, 0);
   CArray rtn = ptr_to_carray(rtn_ptr);
   for(int i = 0; i < ptr_a->x; i++) {
       for(int j = 0; j < ptr_b->x; j++) {
           if(array_a.array1d[i] == array_b.array1d[j]) {
               rtn.array1d[i] = 1;
           }
       }
   }
}
