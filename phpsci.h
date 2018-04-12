
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
  | Author: Henrique Borba <henrisaviatto@gmail.com>                     |
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

void set_obj_uuid(zval * obj, long uuid);
#endif //PHPSCI_EXT_PHPSCI_H
