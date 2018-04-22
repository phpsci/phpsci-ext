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

#ifndef PHPSCI_EXT_PHPSCI_H
#define PHPSCI_EXT_PHPSCI_H

#define PHP_PHPSCI_EXTNAME "PHPSci"
#define PHP_PHPSCI_VERSION "0.0.1"

#ifdef ZTS
#include "TSRM.h"
#endif
#include "kernel/exceptions.h"
#include "php.h"


static zend_class_entry *phpsci_sc_entry;
static zend_object_handlers phpsci_object_handlers;
static zend_class_entry *phpsci_exception_sc_entry;

extern zend_module_entry phpsci_module_entry;

#define phpext_phpsci_ptr &phpsci_module_entry
#define PHPSCI_THROW(message, code) \
		zend_throw_exception(phpsci_exception_sc_entry, message, (long)code TSRMLS_CC); \
		return;


void RETURN_CARRAY(zval * return_value, int uuid, int x, int y);
void set_obj_uuid(zval * obj, long uuid);
#endif //PHPSCI_EXT_PHPSCI_H
