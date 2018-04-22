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
 * Matrix or vector Frobenius norm.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
frobenius_norm(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    if(IS_2D(ptr_a)) {
        carray_init0d(rtn_ptr);
        CArray carray = ptr_to_carray(ptr_a);
        CArray rtn_carray = ptr_to_carray(rtn_ptr);
        rtn_carray.array0d[0] = LAPACKE_dlange(LAPACK_COL_MAJOR, 'F', ptr_a->x, ptr_a->y, carray.array2d, ptr_a->y);
        rtn_ptr->x = 0;
        rtn_ptr->y = 0;
        return;
    }
    throw_atleast2d_exception("Matrix should be at least 2D");
}

/**
 * 1-norm of the matrix
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
norm_1n(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    if(IS_2D(ptr_a)) {
        carray_init0d(rtn_ptr);
        CArray carray = ptr_to_carray(ptr_a);
        CArray rtn_carray = ptr_to_carray(rtn_ptr);
        rtn_carray.array0d[0] = LAPACKE_dlange(LAPACK_COL_MAJOR, '1', ptr_a->x, ptr_a->y, carray.array2d, ptr_a->y);
        rtn_ptr->x = 0;
        rtn_ptr->y = 0;
        return;
    }
    throw_atleast2d_exception("Matrix should be at least 2D");
}

/**
 * Infinity norm of the matrix
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
inf_norm(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    if(IS_2D(ptr_a)) {
        carray_init0d(rtn_ptr);
        CArray carray = ptr_to_carray(ptr_a);
        CArray rtn_carray = ptr_to_carray(rtn_ptr);
        rtn_carray.array0d[0] = LAPACKE_dlange(LAPACK_COL_MAJOR, 'I', ptr_a->x, ptr_a->y, carray.array2d, ptr_a->y);
        rtn_ptr->x = 0;
        rtn_ptr->y = 0;
        return;
    }
    throw_atleast2d_exception("Matrix should be at least 2D");
}

/**
 * Largest absolute norm of the matrix
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 */
void
largest_absolute_norm(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr)
{
    if(IS_2D(ptr_a)) {
        carray_init0d(rtn_ptr);
        CArray carray = ptr_to_carray(ptr_a);
        CArray rtn_carray = ptr_to_carray(rtn_ptr);
        rtn_carray.array0d[0] = LAPACKE_dlange(LAPACK_COL_MAJOR, 'M', ptr_a->x, ptr_a->y, carray.array2d, ptr_a->y);
        rtn_ptr->x = 0;
        rtn_ptr->y = 0;
        return;
    }
    throw_atleast2d_exception("Matrix should be at least 2D");
}

/**
 * Matrix or vector norm.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr_a
 * @param rtn_ptr
 * @param order
 */
void
norm(MemoryPointer * ptr_a, MemoryPointer * rtn_ptr, char * order)
{
    if(!strcmp("fro", order)) {
        // Frobenius Norm ( normF(A) )
        frobenius_norm(ptr_a, rtn_ptr);
        return;
    }
    if(!strcmp("1", order)) {
        // 1-Norm ( norm1(A) )
        norm_1n(ptr_a, rtn_ptr);
        return;
    }
    if(!strcmp("inf", order)) {
        // Inifnity Norm normI(A)
        inf_norm(ptr_a, rtn_ptr);
        return;
    }
    if(!strcmp("m", order)) {
        // Largest Absolute Norm ( max(abs(Aij)) )
        largest_absolute_norm(ptr_a, rtn_ptr);
        return;
    }
    return;
}


