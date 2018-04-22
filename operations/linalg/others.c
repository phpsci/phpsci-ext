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

#include "others.h"
#include "../../kernel/buffer/memory_manager.h"
#include "../../kernel/memory_pointer/utils.h"
#include "../../kernel/carray/carray.h"
#include "../../kernel/memory_pointer/memory_pointer.h"
#include "../../kernel/exceptions.h"
#include "norms.h"
#include "../linalg.h"
#include "lapacke.h"

/**
 * Compute the determinant of an CArray.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_tr
 */
void
other_determinant(MemoryPointer * ptr_a, MemoryPointer * ptr_rtn)
{
    MemoryPointer temp_ptr;
    int ret, iterator_x;
    double det = -1.0;
    lapack_int ipiv[ptr_a->x];
    COPY_PTR(ptr_a, &temp_ptr);
    if(IS_2D(ptr_a)) {
        CArray target_carray = ptr_to_carray(&temp_ptr);
        ret = LAPACKE_dgetrf(LAPACK_COL_MAJOR, ptr_a->x, ptr_a->y, target_carray.array2d, ptr_a->x, ipiv);
        for(iterator_x = 0; iterator_x < ptr_a->x; ++iterator_x) {
            det *= target_carray.array2d[(iterator_x * ptr_a->x) + iterator_x];
            if(ipiv[iterator_x] != iterator_x) {
                det = -det;
            }
        }
        carray_init0d(ptr_rtn);
        CArray rtn_array = ptr_to_carray(ptr_rtn);
        rtn_array.array0d[0] = det;
        ptr_rtn->y = 0;
        ptr_rtn->x = 0;
        return;
    }
    throw_atleast2d_exception("Matrix must be 2D");
}

/**
 * Compute the condition number of a matrix.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param ptr_rtn
 */
void
other_cond(MemoryPointer * ptr_a, MemoryPointer * ptr_rtn)
{
    MemoryPointer inverse_ptr;
    if(IS_2D(ptr_a)) {
        inv(ptr_a, &inverse_ptr);
        norm(&inverse_ptr, ptr_rtn, "fro");
        return;
    }
    throw_atleast2d_exception("Matrix must be at least 2-D");
}
