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

#include "exceptions.h"
#include "php_array.h"
#include "php.h"
#include "Zend/zend_exceptions.h"

#define COULD_NOT_BROADCAST_EXCEPTION 5000

static zend_class_entry * phpsci_ce_CArrayBroadcastException;

static const zend_function_entry phpsci_ce_CArrayBroadcastException_methods[] = {
        PHP_FE_END
};

/**
 * Initialize Exception Classes
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void init_exception_objects()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "CArrayBroadcastException", phpsci_ce_CArrayBroadcastException_methods);
    phpsci_ce_CArrayBroadcastException = zend_register_internal_class_ex(&ce, zend_ce_exception);
}

/**
 * Throw CArrayBroadcastException
 *
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void throw_could_not_broadcast_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayBroadcastException, COULD_NOT_BROADCAST_EXCEPTION, "%s", msg);
}
