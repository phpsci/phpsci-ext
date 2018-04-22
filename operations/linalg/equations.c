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

#include "equations.h"
#include "../../kernel/buffer/memory_manager.h"
#include "../../kernel/memory_pointer/utils.h"
#include "../../kernel/carray/carray.h"
#include "../../kernel/memory_pointer/memory_pointer.h"
#include "../../kernel/exceptions.h"
#include "../transformations.h"
#include "lapacke.h"

/**
 * Solve a linear matrix equation, or system of linear scalar equations.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_b
 * @param rtn_ptr
 */
void
equation_solve(MemoryPointer * ptr_a, MemoryPointer * ptr_b, MemoryPointer * rtn_ptr)
{
    int info;
    lapack_int ipiv[ptr_a->x];
    if(IS_2D(ptr_a) && IS_2D(ptr_b) && carray_shapecmp(ptr_a, ptr_b)) {
        CArray carray_a = ptr_to_carray(ptr_a);
        COPY_PTR(ptr_b, rtn_ptr);
        CArray rtn_arr = ptr_to_carray(rtn_ptr);
        info = LAPACKE_dgesv(LAPACK_COL_MAJOR,
                             ptr_a->x,
                             ptr_b->y,
                             carray_a.array2d,
                             ptr_a->y,
                             ipiv,
                             rtn_arr.array2d,
                             ptr_b->y
        );
        return;
    }
    throw_atleast2d_exception("Matrices provided must be 2-D");
}
