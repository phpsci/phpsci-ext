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
#include "random.h"
#include "../phpsci.h"
#include "../kernel/carray.h"
#include <math.h>
#include <stdlib.h>


/**
 * CArray samples from a standard Normal distribution
 *
 * @author Henrique Borba
 * @param ptr
 */
void standard_normal(MemoryPointer * ptr, int seed, int x, int y)
{
    int i, j;
    srand(time(NULL)*seed);
    if(x > 0 && y == 0) {
        carray_init1d(x, ptr);
        CArray new_arr = ptr_to_carray(ptr);
        for(i = 0; i < x; i++) {
            new_arr.array1d[i] = _randn(0.f, 1.0f);
        }
    }
    if(x > 0 && y > 0) {
        carray_init(x, y, ptr);
        CArray new_arr = ptr_to_carray(ptr);
        for(i = 0; i < x; i++) {
            for(j = 0; j < y; j++)
                new_arr.array2d[i][j] = _randn(0.f, 1.0f);
        }
    }
}


/**
 * Return a sample CArray from the “standard normal” distribution.
 *
 * Based on: https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
 *
 * @author Henrique Borba
 * @param mu
 * @param sigma
 */
float _randn (double mu, double sigma)
{
    double U1, U2, W, mult;
    static float X1, X2;
    static int call = 0;

    if (call == 1)
    {
        call = !call;
        return (float) (mu + sigma * (float) X2);
    }

    do
    {
        U1 = -1 + ((float) rand () / RAND_MAX) * 2;
        U2 = -1 + ((float) rand () / RAND_MAX) * 2;
        W = pow (U1, 2) + pow (U2, 2);
    }
    while (W >= 1 || W == 0);

    mult = sqrt ((-2 * log (W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;

    call = !call;

    return (mu + sigma * (float) X1);
}