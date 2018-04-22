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
#include "utils.h"
#include "../buffer/memory_manager.h"

/**
 * Check if shape of A is equal to shape of B
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
int
carray_shapecmp(MemoryPointer * ptr_a, MemoryPointer * ptr_b)
{
    CArray carray_a = ptr_to_carray(ptr_a);
    CArray carray_b = ptr_to_carray(ptr_b);
    if(ptr_a->x == ptr_b->x && ptr_b->x == ptr_b->y) {
        return 1;
    }
    return 0;
}

/**
 * Get CArray dimensions based on X and Y
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int
GET_DIM(MemoryPointer * ptr)
{
    if(ptr->x == 0 && ptr->y == 0)
        return 0;
    if(ptr->x > 0 && ptr->y == 0)
        return 1;
    if(ptr->x > 0 && ptr->y > 0)
        return 2;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int
IS_0D(MemoryPointer * ptr)
{
    if(ptr->x == 0 && ptr->y == 0)
        return 1;
    return 0;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int
IS_1D(MemoryPointer * ptr)
{
    if(ptr->x > 0 && ptr->y == 0)
        return 1;
    return 0;
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param x
 * @param y
 */
int
IS_2D(MemoryPointer * ptr)
{
    if(ptr->x > 0 && ptr->y > 0)
        return 1;
    return 0;
}
