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