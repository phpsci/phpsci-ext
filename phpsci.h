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

#ifndef CARRAY_EXT_PHPSCI_H
#define CARRAY_EXT_PHPSCI_H

#define PHP_CARRAY_EXTNAME "CArray"
#define PHP_CARRAY_VERSION "0.0.1"

#ifdef ZTS
#include "TSRM.h"
#endif
#include "php.h"
#include "Python.h"
#include "zend.h"

/*
 * Make sure our version of Python is recent enough and that it has been
 * built with all of the options that we need.
 */
#if !defined(PY_VERSION_HEX) || PY_VERSION_HEX <= 0x03050000
    #error Sorry, the Python extension requires Python 3.5.0 or later.
#endif
#if !defined(WITH_THREAD)
    #error Sorry, the Python extension requires Python's threading support.
#endif

static zend_class_entry *carray_sc_entry;
static zend_object_handlers carray_object_handlers;
static zend_class_entry *carray_exception_sc_entry;
static zend_class_entry *carray_iterator_sc_entry;

extern zend_module_entry carray_module_entry;

PyThreadState * GTS();

#define phpext_carray_ptr &carray_module_entry
#endif //PHPSCI_EXT_PHPSCI_H
