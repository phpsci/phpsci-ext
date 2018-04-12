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
#include "ranges.h"
#include "../phpsci.h"
#include "transformations.h"
#include "../kernel/carray.h"
#include <math.h>
#include "zend_exceptions.h"

/**
 * Return evenly spaced values within a given interval.
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void arange(MemoryPointer * new_ptr, float start, float stop, float step, int * width) {
    int i, j;
    if(stop < start || stop == start) {
        PHPSCI_THROW("[PHPSCI CARRAY] Stop < Start in arange() call", 1274);
    }
    if(stop > start) {
        *width = ceil(((stop-start) / step));
        carray_init1d(*width, new_ptr);
        CArray new_array = ptr_to_carray(new_ptr);
        for(i = 0; i < *width; ++i) {
            new_array.array1d[i] = start + (i * step);
        }
    }
}

/**
 * Return CArray 1D with evenly spaced numbers over a specified interval.
 *
 * @author  Henrique Borba <henrique.borba.dev>
 * @param ptr   MemoryPointer
 * @param start The starting value of the sequence.
 * @param stop  The end value of the sequence
 * @param num   Number of samples to generate.
 * @return void
 */
void linspace(MemoryPointer * ptr, float start, float stop, int num)
{
    float step = (stop - start) / (num - 1);
    int i;
    carray_init1d(num, ptr);
    CArray new_array = ptr_to_carray(ptr);
    for(i = 0;i < num; i++) {
        new_array.array1d[i] = start + (i * step);
    }
    new_array.array1d[num - 1] = stop;
}

/**
 * Return numbers spaced evenly on a log scale.
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void logspace(MemoryPointer * ptr, float start, float stop, int num, float base) {
    int i;
    float step = (stop - start)/(num - 1);
    carray_init1d(num, ptr);
    CArray new_array = ptr_to_carray(ptr);
    for(i = 0; i < num; ++i) {
        new_array.array1d[i] = pow(base, (start + (i*step)));
    }
}
