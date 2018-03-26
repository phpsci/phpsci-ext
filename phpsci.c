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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "phpsci.h"
#include "carray/initializers.h"
#include "carray/linalg.h"
#include "carray/transformations.h"
#include "carray/ranges.h"
#include "kernel/memory_manager.h"
#include "php.h"



/**
 * PHPSci Constructor
 */
PHP_METHOD(CArray, __construct)
{
    array_init(return_value);
}

/**
 *
 * INITIALIZERS SECTION
 *
 */
PHP_METHOD(CArray, identity)
{
    long m;
    ZEND_PARSE_PARAMETERS_START(1,1)
        Z_PARAM_LONG(m)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    carray_init((int)m, (int)m, &ptr);
    CArray arr = ptr_to_carray(&ptr);
    identity(&arr, (int)m);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr.uuid);
}
PHP_METHOD(CArray, zeros)
{
    long x, y;
    ZEND_PARSE_PARAMETERS_START(2,2)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();

    MemoryPointer ptr;
    carray_init((int)x, (int)y, &ptr);
    CArray arr = ptr_to_carray(&ptr);
    zeros(&arr, (int)x, (int)y);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr.uuid);
}
PHP_METHOD(CArray, fromArray)
{
    zval * array;
    int a_rows, a_cols;

    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_ARRAY(array)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ptr.uuid = NULL;
    array_to_carray_ptr(&ptr, array, &a_rows, &a_cols);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "rows", sizeof("rows") - 1, a_rows);
    zend_update_property_long(phpsci_sc_entry, return_value, "cols", sizeof("cols") - 1, a_cols);
}
PHP_METHOD(CArray, destroy)
{
    long uuid, rows, cols;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(rows)
        Z_PARAM_LONG(cols)
    ZEND_PARSE_PARAMETERS_END();

    if(cols > 0 && rows > 0) {
        destroy_carray((int)uuid, (int) rows, (int) cols);
    }
}
PHP_METHOD(CArray, transpose)
{
    long uuid, rows, cols;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(rows)
        Z_PARAM_LONG(cols)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    MemoryPointer rtn;
    ptr.uuid = (int)uuid;
    transpose(&rtn, &ptr, (int)rows, (int)cols);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, rtn.uuid);
}
PHP_METHOD(CArray, toArray)
{
    long uuid, rows, cols;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(rows)
        Z_PARAM_LONG(cols)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ptr.uuid = (int)uuid;
    CArray arr = ptr_to_carray(&ptr);
    carray_to_array(arr, return_value, rows, cols);
}
PHP_METHOD(CArray, toDouble)
{
    long uuid;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(uuid)
    ZEND_PARSE_PARAMETERS_END();

    MemoryPointer ptr;
    ptr.uuid = (int)uuid;
    CArray arr = ptr_to_carray(&ptr);
    ZVAL_DOUBLE(return_value, (double)arr.array0d[0]);
}
PHP_METHOD(CArray, matmul)
{
    long a_uuid, a_rows, a_cols, b_uuid, b_rows, b_cols;
    MemoryPointer a_ptr, b_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(5, 5)
        Z_PARAM_LONG(a_uuid)
        Z_PARAM_LONG(a_rows)
        Z_PARAM_LONG(a_cols)
        Z_PARAM_LONG(b_uuid)
        Z_PARAM_LONG(b_cols)
    ZEND_PARSE_PARAMETERS_END();
    a_ptr.uuid = (int)a_uuid;
    b_ptr.uuid = (int)b_uuid;
    matmul(&rtn_ptr, (int)a_rows, (int)a_cols, &a_ptr, (int)b_cols, &b_ptr);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, rtn_ptr.uuid);
}
PHP_METHOD(CArray, arange)
{
    double start, stop, step;
    int width;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_DOUBLE(step)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer * ptr;
    arange(ptr, start, stop, step, &width);
    object_init(return_value);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr->uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "rows", sizeof("rows") - 1, width);
    zend_update_property_long(phpsci_sc_entry, return_value, "cols", sizeof("cols") - 1, 0);
}

/**
 * CLASS METHODS
 */
static zend_function_entry phpsci_class_methods[] =
{
   PHP_ME(CArray, __construct, NULL, ZEND_ACC_PUBLIC)
   // RANGES SECTION
   PHP_ME(CArray, arange, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   // TRANSFORMATIONS SECTION
   PHP_ME(CArray, transpose, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   // PRODUCTS SECTION
   PHP_ME(CArray, matmul, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   // CARRAY MEMORY MANAGEMENT SECTION
   PHP_ME(CArray, destroy, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   // INITIALIZERS SECTION
   PHP_ME(CArray, identity, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, zeros, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   // CONVERT SECTION
   PHP_ME(CArray, toArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, fromArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, toDouble, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   { NULL, NULL, NULL }
};

zend_function_entry phpsci_functions[] = {
        {NULL, NULL, NULL}
};

/**
 * MINIT
 */
static PHP_MINIT_FUNCTION(phpsci)
{
    zend_class_entry ce;
    memcpy(&phpsci_object_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));

    INIT_CLASS_ENTRY(ce, "CArray", phpsci_class_methods);
    ce.create_object = NULL;
    phpsci_object_handlers.clone_obj = NULL;
    phpsci_sc_entry = zend_register_internal_class(&ce TSRMLS_CC);

    return SUCCESS;
}

/**
 * MINFO
 */
static PHP_MINFO_FUNCTION(phpsci)
{
    php_info_print_table_start();
    php_info_print_table_row(2, "PHPSci support", "enabled");
    php_info_print_table_row(2, "PHPSci version", PHP_PHPSCI_VERSION);
    php_info_print_table_end();
}

/**
 * MSHUTDOWN
 */
static PHP_MSHUTDOWN_FUNCTION(phpsci)
{
    UNREGISTER_INI_ENTRIES();
    return SUCCESS;
}

zend_module_entry phpsci_module_entry = {
        STANDARD_MODULE_HEADER,
        PHP_PHPSCI_EXTNAME,
        phpsci_functions,				/* Functions */
        PHP_MINIT(phpsci),				/* MINIT */
        PHP_MSHUTDOWN(phpsci),			/* MSHUTDOWN */
        NULL,						    /* RINIT */
        NULL,						    /* RSHUTDOWN */
        PHP_MINFO(phpsci),				/* MINFO */
        PHP_PHPSCI_VERSION,				/* version */
        STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_PHPSCI
ZEND_GET_MODULE(phpsci)
#endif /* COMPILE_DL_PHPSCI */