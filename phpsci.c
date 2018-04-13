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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "phpsci.h"
#include "carray/initializers.h"
#include "carray/linalg.h"
#include "carray/basic_operations.h"
#include "carray/transformations.h"
#include "carray/random.h"
#include "carray/ranges.h"
#include "carray/arithmetic.h"
#include "kernel/carray_printer.h"
#include "kernel/memory_manager.h"
#include "php.h"
#include "ext/standard/info.h"

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 * @param obj
 * @param uuid
 */
void set_obj_uuid(zval * obj, long uuid)
{
    zend_update_property_long(phpsci_sc_entry, obj, "uuid", sizeof("uuid") - 1, uuid);
}

/**
 * @author Henrique Borba <henrique.borba.dev@gmail.com>
 */
void RETURN_CARRAY(zval * return_value, int uuid, int x, int y)
{
    object_init_ex(return_value, phpsci_sc_entry);
    zend_update_property_long(phpsci_sc_entry, return_value, "uuid", sizeof("uuid") - 1, uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, x);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, y);
}



PHP_METHOD(CArray, __construct)
{
    long uuid, x, y;
    ZEND_PARSE_PARAMETERS_START(3,3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    zval * obj = getThis();
    set_obj_uuid(obj, uuid);
    zend_update_property_long(phpsci_sc_entry, obj, "x", sizeof("x") - 1, x);
    zend_update_property_long(phpsci_sc_entry, obj, "y", sizeof("y") - 1, y);
}
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
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, m);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, m);
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
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, x);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, y);
}
PHP_METHOD(CArray, standard_normal)
{
    long x, y, seed;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_LONG(seed)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer * ptr;
    if(y > 0) {
        standard_normal(ptr,(int)seed, (int)x, (int)y);
        object_init_ex(return_value, phpsci_sc_entry);
        set_obj_uuid(return_value, ptr->uuid);
        zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, x);
        zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, y);
        return;
    }
    if(y == 0) {
        standard_normal(ptr,(int)seed, (int)x, 0);
        object_init_ex(return_value, phpsci_sc_entry);
        set_obj_uuid(return_value, ptr->uuid);
        zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, x);
        zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
        return;
    }
}
PHP_METHOD(CArray, fromArray)
{
    zval * array;
    int a_rows, a_cols;

    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_ARRAY(array)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ptr.uuid = UNINITIALIZED;
    array_to_carray_ptr(&ptr, array, &a_rows, &a_cols);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, a_rows);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, a_cols);
}
PHP_METHOD(CArray, destroy)
{
    zval * obj;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer target_ptr;
    OBJ_TO_PTR(obj, &target_ptr);
    destroy_carray(&target_ptr);
}
PHP_METHOD(CArray, transpose)
{
    zval * obj;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    MemoryPointer rtn;
    OBJ_TO_PTR(obj, &ptr);
    transpose(&rtn, &ptr, (int)ptr.x, (int)ptr.y);
    RETURN_CARRAY(return_value, rtn.uuid, ptr.y, ptr.x);
}
PHP_METHOD(CArray, print_r) {
    long uuid, x, y;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ptr.uuid = (int)uuid;
    print_carray(&ptr, x, y);
}
PHP_METHOD(CArray, toArray)
{
    int rows, cols;
    zval * a, rv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ptr.uuid = (int)zval_get_long(zend_read_property(phpsci_sc_entry, a, "uuid", sizeof("uuid") - 1, 1, &rv));
    rows = (int)zval_get_long(zend_read_property(phpsci_sc_entry, a, "x", sizeof("x") - 1, 1, &rv));
    cols = (int)zval_get_long(zend_read_property(phpsci_sc_entry, a, "y", sizeof("y") - 1, 1, &rv));
    CArray arr = ptr_to_carray(&ptr);
    carray_to_array(arr, return_value, rows, cols);
}
PHP_METHOD(CArray, linspace)
{
    double start, stop;
    long num;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_LONG(num)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    linspace(&ptr, (float)start, (float)stop, (float)num);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, num);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
}
PHP_METHOD(CArray, logspace)
{
    double start, stop, base;
    long num;
    ZEND_PARSE_PARAMETERS_START(4, 4)
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_LONG(num)
        Z_PARAM_DOUBLE(base)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    logspace(&ptr, (float)start, (float)stop, num, (float)base);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, num);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
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
PHP_METHOD(CArray, sum)
{
    long axis;
    int size_x, size_y;
    zval * a;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer a_ptr;
    OBJ_TO_PTR(a, &a_ptr);
    MemoryPointer target_ptr;
    if (ZEND_NUM_ARGS() == 1) {
        sum_noaxis(&a_ptr, &target_ptr, a_ptr.x, a_ptr.y);
        size_x = 0;
        size_y = 0;
    }
    if (ZEND_NUM_ARGS() == 2) {
        sum_axis(&a_ptr, &target_ptr, a_ptr.x, a_ptr.y, (int)axis, &size_x, &size_y);
    }
    RETURN_CARRAY(return_value, target_ptr.uuid, size_x, size_y);
}
PHP_METHOD(CArray, inner)
{
    long a_uuid, a_x, a_y, b_uuid, b_x, b_y;
    MemoryPointer a_ptr, b_ptr, rtn_ptr;
    int rtn_x = 0, rtn_y = 0;
    zval * a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &a_ptr);
    OBJ_TO_PTR(b, &b_ptr);
    inner(&rtn_x, &rtn_y, &rtn_ptr, (int)a_ptr.x, (int)a_ptr.y, &a_ptr, (int)b_ptr.x, (int)b_ptr.y, &b_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, rtn_x, rtn_y);
}
PHP_METHOD(CArray, matmul)
{
    MemoryPointer a_ptr, b_ptr, rtn_ptr;
    zval * a, * b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &a_ptr);
    OBJ_TO_PTR(b, &b_ptr);
    matmul(&rtn_ptr, (int)a_ptr.x, (int)a_ptr.y, &a_ptr, (int)b_ptr.y, &b_ptr);
    if(a_ptr.y > 0) {
        RETURN_CARRAY(return_value, rtn_ptr.uuid, a_ptr.x, b_ptr.y);
        return;
    }
    if(a_ptr.y == 0) {
        RETURN_CARRAY(return_value, rtn_ptr.uuid, a_ptr.x, a_ptr.y);
        return;
    }
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
    MemoryPointer ptr;
    arange(&ptr, start, stop, step, &width);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, width);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
}
PHP_METHOD(CArray, add)
{
    zval * a, * b;
    int  size_x, size_y;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer rtn_ptr;
    MemoryPointer ptr_a;
    MemoryPointer ptr_b;
    OBJ_TO_PTR(a, &ptr_a);
    OBJ_TO_PTR(b, &ptr_b);
    add(&ptr_a, (int)ptr_a.x, (int)ptr_a.y, &ptr_b, (int)ptr_b.x, (int)ptr_b.y, &rtn_ptr, &size_x, &size_y);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, size_x, size_y);
}
PHP_METHOD(CArray, fromDouble)
{
    double input;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_DOUBLE(input)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    double_to_carray(input, &ptr);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, 0);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
}

/**
 * CLASS METHODS
 */
static zend_function_entry phpsci_class_methods[] =
{
   // PHP_ME(CArray, __construct, NULL, ZEND_ACC_PUBLIC)
   // RANGES SECTION
   PHP_ME(CArray, arange, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, linspace, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, logspace, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // ARITHMETIC
   PHP_ME(CArray, add, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // TRANSFORMATIONS SECTION
   PHP_ME(CArray, transpose, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // PRODUCTS SECTION
   PHP_ME(CArray, matmul, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, inner, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // CARRAY MEMORY MANAGEMENT SECTION
   PHP_ME(CArray, destroy, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // INITIALIZERS SECTION
   PHP_ME(CArray, identity, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, zeros, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // BASIC OPERATIONS
   PHP_ME(CArray, sum, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // CONVERT SECTION
   PHP_ME(CArray, toArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, fromArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, toDouble, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, fromDouble, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // VISUALIZATION
   PHP_ME(CArray, print_r, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // RANDOM SECTION
   PHP_ME(CArray, standard_normal, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
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
