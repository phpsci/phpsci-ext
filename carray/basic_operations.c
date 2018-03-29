/*
  +----------------------------------------------------------------------+
  | PHP Version 7 | PHPSci                                               |
  +----------------------------------------------------------------------+
  | Copyright (c) 2018 Henrique Borba                                    |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author: Henrique Borba <henrique.borba.dev@gmail.com>                |
  +----------------------------------------------------------------------+
*/

#include "basic_operations.h"
#include "../phpsci.h"
#include "../kernel/carray.h"

/**
 * Sum of CArray elements.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param x
 * @param y
 * @param axis
 */
void sum_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y) {
    int i, j;
    carray_init0d(target_ptr);
    CArray rtn_arr = ptr_to_carray(target_ptr);
    CArray new_arr = ptr_to_carray(ptr);
    rtn_arr.array0d[0] = 0;
    if(x > 0 && y > 0) {
        for(i = 0; i < x; i++) {
            for(j = 0; j < y; j++) {
                rtn_arr.array0d[0] += new_arr.array2d[i][j];
            }
        }
    }
    if(x > 0 && y == 0) {
        for(i = 0; i < x; i++) {
            rtn_arr.array0d[0] += new_arr.array1d[i];
        }
    }
}
/**
 * Sum of CArray elements over a given axis.
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param ptr
 * @param x
 * @param y
 * @param axis
 */
void sum_axis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y, int axis) {


}

void sub_noaxis(MemoryPointer * ptr, MemoryPointer * target_ptr, int x, int y) {
    int i, j;

    carray_init0d(target_ptr);
    
    CArray rtn_arr = ptr_to_carray(target_ptr);
    CArray new_arr = ptr_to_carray(ptr);
    
    rtn_arr.array0d[0] = 0;
    
    if(x > 0 && y > 0) {
        for(i = 0; i < x; ++i) {
            for(j = 0; j < y; ++j) {
                rtn_arr.array0d[0] -= new_arr.array2d[i][j];
            }
        }

        return;
    }
    
    if(x > 0 && y == 0) {
        for(i = 0; i < x; ++i) {
            rtn_arr.array0d[0] -= new_arr.array1d[i];
        }

        return;
    }
}
