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

#include "initializers.h"
#include "../phpsci.h"

/**
 * Create 2D Identity CArray with shape (m,m)
 *
 * @author Henrique Borba <henrique.borba.dev>
 */
void identity(CArray * carray, int m) {
    int i, j;
    for(i = 0; i < m; i++) {
        for(j = 0; j < m; j++) {
            carray->array2d[i][j] = j == i ? 1.0 : 0.0;
        }
    }
}