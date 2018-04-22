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
#include "ranges.h"
#include "../phpsci.h"
#include "transformations.h"
#include "../kernel/carray/carray.h"
#include <math.h>
#include "zend_exceptions.h"

/**
 * Return evenly spaced values within a given interval.
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void arange(MemoryPointer * new_ptr, double start, double stop, double step, int * width) {
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
void linspace(MemoryPointer * ptr, double start, double stop, int num)
{
    double step = (stop - start) / (num - 1);
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
void logspace(MemoryPointer * ptr, double start, double stop, int num, double base) {
    int i;
    double step = (stop - start)/(num - 1);
    carray_init1d(num, ptr);
    CArray new_array = ptr_to_carray(ptr);
    for(i = 0; i < num; ++i) {
        new_array.array1d[i] = pow(base, (start + (i*step)));
    }
}
