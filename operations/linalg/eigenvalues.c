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
#include "norms.h"
#include "../../kernel/buffer/memory_manager.h"
#include "../../kernel/memory_pointer/utils.h"
#include "../../kernel/carray/carray.h"
#include "../../kernel/memory_pointer/memory_pointer.h"
#include "../../kernel/exceptions.h"
#include "lapacke.h"

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
eig(MemoryPointer * ptr_a, MemoryPointer * rtn_eigvalues_ptr, MemoryPointer * rtn_eigvectors_ptr)
{
    MemoryPointer wr_ptr, wi_ptr, vl_ptr, vr_ptr;
    carray_init1d(ptr_a->x, &wr_ptr);
    carray_init1d(ptr_a->x, &wi_ptr);
    carray_init(ptr_a->x, ptr_a->x, &vl_ptr);
    carray_init(ptr_a->x, ptr_a->x, &vr_ptr);
    CArray wr_carray = ptr_to_carray(&wr_ptr);
    CArray wi_carray = ptr_to_carray(&wi_ptr);
    CArray vl_carray = ptr_to_carray(&vl_ptr);
    CArray vr_carray = ptr_to_carray(&vr_ptr);
    if(IS_2D(ptr_a) && IS_SQUARE(ptr_a)) {
        CArray target_array = ptr_to_carray(ptr_a);
        LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V',
                      ptr_a->x, target_array.array2d,
                      ptr_a->x, wr_carray.array1d,
                      wi_carray.array1d,
                      vl_carray.array2d,
                      ptr_a->x,
                      vr_carray.array2d,
                      ptr_a->x
        );
        COPY_PTR(&wr_ptr, rtn_eigvalues_ptr);
        COPY_PTR(&vr_ptr, rtn_eigvectors_ptr);
        return;
    }
    throw_atleast2d_exception("Matrix must be 2-D and with square shape (N, N)");
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
eigvals(MemoryPointer * ptr_a, MemoryPointer * rtn_eigvalues_ptr)
{
    MemoryPointer wr_ptr, wi_ptr, vl_ptr, vr_ptr;
    carray_init1d(ptr_a->x, &wr_ptr);
    carray_init1d(ptr_a->x, &wi_ptr);
    carray_init(ptr_a->x, ptr_a->x, &vl_ptr);
    carray_init(ptr_a->x, ptr_a->x, &vr_ptr);
    CArray wr_carray = ptr_to_carray(&wr_ptr);
    CArray wi_carray = ptr_to_carray(&wi_ptr);
    CArray vl_carray = ptr_to_carray(&vl_ptr);
    CArray vr_carray = ptr_to_carray(&vr_ptr);
    if(IS_2D(ptr_a) && IS_SQUARE(ptr_a)) {
        CArray target_array = ptr_to_carray(ptr_a);
        LAPACKE_dgeev(LAPACK_COL_MAJOR, 'V', 'V',
                      ptr_a->x, target_array.array2d,
                      ptr_a->x, wr_carray.array1d,
                      wi_carray.array1d,
                      vl_carray.array2d,
                      ptr_a->x,
                      vr_carray.array2d,
                      ptr_a->x
        );
        COPY_PTR(&wr_ptr, rtn_eigvalues_ptr);
        return;
    }
    throw_atleast2d_exception("Matrix must be 2-D and with square shape (N, N)");
}