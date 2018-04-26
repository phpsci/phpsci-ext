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
#include "operations/initializers.h"
#include "operations/linalg.h"
#include "operations/basic_operations.h"
#include "operations/transformations.h"
#include "operations/random.h"
#include "operations/ranges.h"
#include "operations/arithmetic.h"
#include "operations/logarithms.h"
#include "operations/exponents.h"
#include "operations/trigonometric.h"
#include "operations/hyperbolic.h"
#include "operations/magic_properties.h"
#include "operations/linalg/norms.h"
#include "operations/linalg/others.h"
#include "operations/linalg/eigenvalues.h"
#include "operations/linalg/equations.h"
#include "kernel/carray/utils/carray_printer.h"
#include "kernel/buffer/memory_manager.h"
#include "kernel/php/php_array.h"
#include "kernel/exceptions.h"
#include "kernel/memory_pointer/utils.h"
#include "php.h"
#include "ext/standard/info.h"
#include "Zend/zend_interfaces.h"



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
    zend_long uuid, x, y;
    ZEND_PARSE_PARAMETERS_START(3,3)
        Z_PARAM_LONG(uuid)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    zval * obj = getThis();
    set_obj_uuid(obj, (int)uuid);
    zend_update_property_long(phpsci_sc_entry, obj, "x", sizeof("x") - 1, x);
    zend_update_property_long(phpsci_sc_entry, obj, "y", sizeof("y") - 1, y);
}
PHP_METHOD(CArray, identity)
{
    zend_long m;
    ZEND_PARSE_PARAMETERS_START(1,1)
        Z_PARAM_LONG(m)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    carray_init((int)m, (int)m, &ptr);
    CArray arr = ptr_to_carray(&ptr);
    identity(&arr, (int)m);
    RETURN_CARRAY(return_value, ptr.uuid, (int)m, (int)m);
}
PHP_METHOD(CArray, zeros)
{
    int x, y;
    zval * a;
    MemoryPointer ptr;
    Tuple shape;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_TUPLE(a, &shape);
    if(shape.size == 1) {
        x = shape.t[0];
        y = 0;
    }
    if(shape.size == 2) {
        x = shape.t[0];
        y = shape.t[1];
    }
    zeros(&ptr, x, y);
    RETURN_CARRAY(return_value, ptr.uuid, x, y);
}
PHP_METHOD(CArray, ones)
{
    zend_long x, y;
    ZEND_PARSE_PARAMETERS_START(2,2)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    ones(&ptr, (int)x, (int)y);
    RETURN_CARRAY(return_value, ptr.uuid, (int)x, (int)y);
}
PHP_METHOD(CArray, full)
{
    zend_long x, y;
    double num;
    ZEND_PARSE_PARAMETERS_START(3,3)
        Z_PARAM_DOUBLE(num)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    full(&ptr, (int)x, (int)y, num);
    RETURN_CARRAY(return_value, ptr.uuid, (int)x, (int)y);
}
PHP_METHOD(CArray, full_like)
{
    zval * obj;
    double num;
    MemoryPointer ptr;
    MemoryPointer new_array_ptr;
    ZEND_PARSE_PARAMETERS_START(2,2)
        Z_PARAM_OBJECT(obj)
        Z_PARAM_DOUBLE(num)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &ptr);
    full(&new_array_ptr, ptr.x, ptr.y, num);
    RETURN_CARRAY(return_value, new_array_ptr.uuid, ptr.x, ptr.y);
}
PHP_METHOD(CArray, ones_like)
{
    zval * obj;
    MemoryPointer ptr;
    MemoryPointer new_array_ptr;
    ZEND_PARSE_PARAMETERS_START(1,1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &ptr);
    ones(&new_array_ptr, ptr.x, ptr.y);
    RETURN_CARRAY(return_value, new_array_ptr.uuid, ptr.x, ptr.y);
}
PHP_METHOD(CArray, zeros_like)
{
    zval * obj;
    MemoryPointer ptr;
    MemoryPointer new_array_ptr;
    ZEND_PARSE_PARAMETERS_START(1,1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &ptr);
    zeros(&new_array_ptr, ptr.x, ptr.y);
    RETURN_CARRAY(return_value, new_array_ptr.uuid, ptr.x, ptr.y);
}
PHP_METHOD(CArray, standard_normal)
{
    zend_long x, y, seed = 1;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_LONG(x)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    standard_normal(&ptr,(int)seed, (int)x, 0);
    object_init_ex(return_value, phpsci_sc_entry);
    set_obj_uuid(return_value, ptr.uuid);
    zend_update_property_long(phpsci_sc_entry, return_value, "x", sizeof("x") - 1, x);
    zend_update_property_long(phpsci_sc_entry, return_value, "y", sizeof("y") - 1, 0);
    return;
}
PHP_METHOD(CArray, atleast_1d)
{
    zval * obj;
    MemoryPointer target_ptr;
    MemoryPointer return_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &target_ptr);
    atleast_1d(&return_ptr, &target_ptr);
    RETURN_CARRAY(return_value, return_ptr.uuid, return_ptr.x, return_ptr.y);
}
PHP_METHOD(CArray, eig)
{
    zval * obj, eigvals_obj, eigvectors_obj;
    MemoryPointer target_ptr, rtn_eigvalues_ptr, rtn_eigvectors_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &target_ptr);
    eig(&target_ptr, &rtn_eigvalues_ptr, &rtn_eigvectors_ptr);
    array_init(return_value);
    RETURN_CARRAY(&eigvectors_obj, rtn_eigvectors_ptr.uuid, target_ptr.x, target_ptr.y);
    RETURN_CARRAY(&eigvals_obj, rtn_eigvalues_ptr.uuid, target_ptr.x, 0);
    add_next_index_zval(return_value, &eigvals_obj);
    add_next_index_zval(return_value, &eigvectors_obj);
}
PHP_METHOD(CArray, eigvals)
{
    zval * obj;
    MemoryPointer target_ptr, rtn_eigvalues_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &target_ptr);
    eigvals(&target_ptr, &rtn_eigvalues_ptr);
    RETURN_CARRAY(return_value, rtn_eigvalues_ptr.uuid, target_ptr.x, 0);
}
PHP_METHOD(CArray, atleast_2d)
{
    zval * obj;
    MemoryPointer target_ptr;
    MemoryPointer return_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(obj)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(obj, &target_ptr);
    atleast_2d(&return_ptr, &target_ptr);
    RETURN_CARRAY(return_value, return_ptr.uuid, return_ptr.x, return_ptr.y);
}
PHP_METHOD(CArray, flatten)
{
    zval * obj = getThis();
    MemoryPointer target_ptr;
    MemoryPointer return_ptr;
    OBJ_TO_PTR(obj, &target_ptr);
    flatten(&return_ptr, &target_ptr);
    RETURN_CARRAY(return_value, return_ptr.uuid, return_ptr.x, return_ptr.y);
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
    RETURN_CARRAY(return_value, ptr.uuid, a_rows, a_cols);
}
PHP_METHOD(CArray, __destruct)
{
    zval * obj = getThis();
    MemoryPointer target_ptr;
    OBJ_TO_PTR(obj, &target_ptr);
    //destroy_carray(&target_ptr);
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
    transpose(&rtn, &ptr);
    RETURN_CARRAY(return_value, rtn.uuid, ptr.y, ptr.x);
}
PHP_METHOD(CArray, eye)
{
    zend_long x, y, k;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_LONG(x)
        Z_PARAM_LONG(y)
        Z_PARAM_LONG(k)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    eye(&ptr, (int)x, (int)y, (int)k);
    RETURN_CARRAY(return_value, (int)ptr.uuid, (int)x, (int)y);
}
PHP_METHOD(CArray, print_r) {
    zval * a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    OBJ_TO_PTR(a, &ptr);
    print_carray(&ptr, ptr.x, ptr.y);
}
PHP_METHOD(CArray, toArray)
{
    int rows, cols;
    zval * a, rv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    OBJ_TO_PTR(a, &ptr);
    CArray arr = ptr_to_carray(&ptr);
    carray_to_array(arr, return_value, ptr.x, ptr.y);
}
PHP_METHOD(CArray, linspace)
{
    double start, stop;
    zend_long num;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_LONG(num)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    linspace(&ptr, (double)start, (double)stop, (double)num);
    RETURN_CARRAY(return_value, ptr.uuid, num, 0);
}
PHP_METHOD(CArray, logspace)
{
    double start, stop, base;
    zend_long num;
    ZEND_PARSE_PARAMETERS_START(4, 4)
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_LONG(num)
        Z_PARAM_DOUBLE(base)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    logspace(&ptr, (double)start, (double)stop, num, (double)base);
    RETURN_CARRAY(return_value, ptr.uuid, num, 0);
}
PHP_METHOD(CArray, toDouble)
{
    zval * a;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr);
    CArray arr = ptr_to_carray(&ptr);
    ZVAL_DOUBLE(return_value, (double)arr.array0d[0]);
}
PHP_METHOD(CArray, exp)
{
    zend_long axis;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer a_ptr;
    OBJ_TO_PTR(a, &a_ptr);
    MemoryPointer target_ptr;
    exponential(&a_ptr, &target_ptr, a_ptr.x, a_ptr.y);
    RETURN_CARRAY(return_value, target_ptr.uuid, a_ptr.x, a_ptr.y);
}
PHP_METHOD(CArray, log)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    natural_log(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, log10)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    base10_log(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, log2)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    base2_log(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, log1p)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    loga1p(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, tan)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    tan_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, sin)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    sin_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, cos)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    cos_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, arccos)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    arccos_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, arcsin)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    arcsin_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, arctan)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    arctan_carray(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, sinh)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    hyperbolic_sinh(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, tanh)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    hyperbolic_tanh(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, cosh)
{
    zval * a;
    MemoryPointer target_ptr, rtn_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &target_ptr);
    hyperbolic_cosh(&target_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, target_ptr.x, target_ptr.y);
}
PHP_METHOD(CArray, sum)
{
    zend_long axis;
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
PHP_METHOD(CArray, outer)
{
    MemoryPointer a_ptr, b_ptr, rtn_ptr;
    zval * a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &a_ptr);
    OBJ_TO_PTR(b, &b_ptr);
    outer(&a_ptr, &b_ptr, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, (a_ptr.x * a_ptr.y), (a_ptr.x * a_ptr.y));
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
    MemoryPointer ptr;
    double start, stop, step;
    int width;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_DOUBLE(stop)
        Z_PARAM_OPTIONAL
        Z_PARAM_DOUBLE(start)
        Z_PARAM_DOUBLE(step)
    ZEND_PARSE_PARAMETERS_END();
    if (ZEND_NUM_ARGS() == 1) {
        start = 0.0;
        step =  1.0;
    }
    arange(&ptr, start, stop, step, &width);
    RETURN_CARRAY(return_value, ptr.uuid, width, 0);
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
    add(&ptr_a, &ptr_b, &rtn_ptr, &size_x, &size_y);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, size_x, size_y);
}
PHP_METHOD(CArray, subtract)
{
    MemoryPointer rtn_ptr, ptr_a, ptr_b;
    zval * a, * b;
    int  size_x, size_y;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();

    OBJ_TO_PTR(a, &ptr_a);
    OBJ_TO_PTR(b, &ptr_b);
    subtract(&ptr_a, &ptr_b, &rtn_ptr, &size_x, &size_y);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, size_x, size_y);
}
PHP_METHOD(CArray, det)
{
    MemoryPointer ptr_a, rtn_ptr;
    zval * a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr_a);
    other_determinant(&ptr_a, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, rtn_ptr.x, rtn_ptr.y);
}
PHP_METHOD(CArray, cond)
{
    MemoryPointer ptr_a, rtn_ptr;
    zval * a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr_a);
    other_cond(&ptr_a, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, 0, 0);
}
PHP_METHOD(CArray, solve)
{
    MemoryPointer ptr_a, ptr_b, rtn_ptr;
    zval * a, * b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OBJECT(b)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr_a);
    OBJ_TO_PTR(b, &ptr_b);
    equation_solve(&ptr_a, &ptr_b, &rtn_ptr);
    RETURN_CARRAY(return_value, rtn_ptr.uuid, ptr_a.x, ptr_b.y);
}
PHP_METHOD(CArray, norm)
{
    char * order_name;
    size_t order_name_len;
    MemoryPointer this_ptr, new_ptr;
    zval * a;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_OBJECT(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(order_name, order_name_len)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &this_ptr);
    if (ZEND_NUM_ARGS() == 1) {
        carray_norm(&this_ptr, &new_ptr, "fro");
    }
    if (ZEND_NUM_ARGS() == 2) {
        carray_norm(&this_ptr, &new_ptr, order_name);
    }
    RETURN_CARRAY(return_value, new_ptr.uuid, new_ptr.x, new_ptr.y);
}
PHP_METHOD(CArray, fromDouble)
{
    double input;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_DOUBLE(input)
    ZEND_PARSE_PARAMETERS_END();
    MemoryPointer ptr;
    double_to_carray(input, &ptr);
    RETURN_CARRAY(return_value, ptr.uuid, 0, 0);
}
PHP_METHOD(CArray, __get)
{
    char * name;
    size_t name_len;
    MemoryPointer this_ptr, new_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STRING(name, name_len)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(getThis(), &this_ptr);
    run_property_or_die(name, return_value, &this_ptr, &new_ptr);
}
// THIS IS REQUIRED BY __get() MAGIC METHOD
ZEND_BEGIN_ARG_INFO_EX(phpsci_get_args, 0, 0, 2)
    ZEND_ARG_INFO(0, name)
ZEND_END_ARG_INFO()

PHP_METHOD(CArray, inv)
{
    zval * a;
    MemoryPointer ptr_a;
    MemoryPointer rtn;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr_a);
    inv(&ptr_a, &rtn);
    RETURN_CARRAY(return_value, rtn.uuid, ptr_a.x, ptr_a.y);
}
PHP_METHOD(CArray, svd)
{
    zval * a, singular_obj, left_obj, right_obj;
    MemoryPointer ptr_a;
    MemoryPointer temp_ptr;
    MemoryPointer singular_ptr;
    MemoryPointer left_ptr;
    MemoryPointer right_ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT(a)
    ZEND_PARSE_PARAMETERS_END();
    OBJ_TO_PTR(a, &ptr_a);
    svd(&ptr_a, &temp_ptr, &singular_ptr, &left_ptr, &right_ptr);
    array_init(return_value);
    RETURN_CARRAY(&singular_obj, singular_ptr.uuid, ptr_a.y, 0);
    RETURN_CARRAY(&left_obj, left_ptr.uuid, ptr_a.x, ptr_a.x);
    RETURN_CARRAY(&right_obj, right_ptr.uuid, ptr_a.y, ptr_a.y);
    add_next_index_zval(return_value, &singular_obj);
    add_next_index_zval(return_value, &left_obj);
    add_next_index_zval(return_value, &right_obj);
}

PHP_METHOD(CArray, offsetExists)
{
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
}
PHP_METHOD(CArray, offsetGet)
{
    MemoryPointer target_ptr, rtn_ptr;
    Tuple index_t;
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
    OBJ_TO_PTR(getThis(), &target_ptr);
    if(Z_TYPE_P(index) == IS_ARRAY) {
        OBJ_TO_TUPLE(index, &index_t);
        if(index_t.size == 2 && IS_2D(&target_ptr)) {
            ZVAL_DOUBLE(return_value, carray_get_value(&target_ptr, &index_t));
        }
        if(index_t.size == 1 && IS_2D(&target_ptr)) {
            carray_get_inner_carray(&target_ptr, &rtn_ptr, index_t);
            RETURN_CARRAY(return_value, rtn_ptr.uuid, rtn_ptr.x, rtn_ptr.y);
        }
        if(index_t.size == 1 && IS_1D(&target_ptr)) {
            ZVAL_DOUBLE(return_value, carray_get_value(&target_ptr, &index_t));
        }
    }
}
ZEND_BEGIN_ARG_INFO_EX(arginfo_array_offsetGet, 0, 0, 1)
    ZEND_ARG_INFO(0, index)
ZEND_END_ARG_INFO()
PHP_METHOD(CArray, offsetSet)
{
    MemoryPointer target_ptr, inner_ptr;
    Tuple index_t;
    zval *index, *value;
    int x, y;
    int dim;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "zz", &index, &value) == FAILURE) {
        return;
    }
    OBJ_TO_PTR(getThis(), &target_ptr);
    if(Z_TYPE_P(index) == IS_ARRAY) {
        OBJ_TO_TUPLE(index, &index_t);
        if(Z_TYPE_P(value) == IS_DOUBLE || Z_TYPE_P(value) == IS_LONG) {
            convert_to_double(value);
            carray_set_value(&target_ptr, &index_t,(int)Z_DVAL_P(value));
            return;
        }
        if(Z_TYPE_P(value) == IS_ARRAY) {
            array_to_carray_ptr(&inner_ptr, value, &x, &y);
            carray_set_inner_carray(&target_ptr, &inner_ptr, index_t);
            return;
        }
        OBJ_TO_PTR(value, &inner_ptr);
        carray_set_inner_carray(&target_ptr, &inner_ptr, index_t);
        return;
    }
}
ZEND_BEGIN_ARG_INFO_EX(arginfo_array_offsetSet, 0, 0, 2)
    ZEND_ARG_INFO(0, index)
    ZEND_ARG_INFO(0, newval)
ZEND_END_ARG_INFO()
PHP_METHOD(CArray, offsetUnset)
{
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
}
PHP_METHOD(CArray, __toString)
{
    zend_string *str;
    zval * a = getThis();
    MemoryPointer ptr;
    str = ZSTR_EMPTY_ALLOC();
    OBJ_TO_PTR(a, &ptr);
    print_carray(&ptr, ptr.x, ptr.y);
    RETURN_STR(str);
}
/**
 * CLASS METHODS
 */
static zend_function_entry phpsci_class_methods[] =
{
   // CARRAY ITERATOR
   PHP_ME(CArray, offsetUnset, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)
   PHP_ME(CArray, offsetSet, arginfo_array_offsetSet, ZEND_ACC_PUBLIC)
   PHP_ME(CArray, offsetGet, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)
   PHP_ME(CArray, offsetExists, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)

   // PHP_ME(CArray, __construct, NULL, ZEND_ACC_PUBLIC)
   // RANGES SECTION
   PHP_ME(CArray, arange, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, linspace, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, logspace, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // ARITHMETIC
   PHP_ME(CArray, add, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, subtract, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // LOGARITHMS
   PHP_ME(CArray, log, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, log10, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, log2, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, log1p, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // TRANSFORMATIONS SECTION
   PHP_ME(CArray, transpose, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, atleast_1d, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, atleast_2d, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, flatten, NULL, ZEND_ACC_PUBLIC)

   // EIGENVALUES
   PHP_ME(CArray, eig, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, eigvals, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // NORMS SECTION
   PHP_ME(CArray, norm, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // OTHERS SECTION
   PHP_ME(CArray, det, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, cond, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // PRODUCTS SECTION
   PHP_ME(CArray, matmul, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, inner, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, outer, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, inv, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, svd, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // EQUATIONS SECTION
   PHP_ME(CArray, solve, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // MAGIC PROPERTIES
   PHP_ME(CArray, __get, phpsci_get_args, ZEND_ACC_PUBLIC)

   // CARRAY MEMORY MANAGEMENT SECTION
   PHP_ME(CArray, __destruct, NULL, ZEND_ACC_PUBLIC)

   // TRIGONOMETRIC
   PHP_ME(CArray, tan, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, cos, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, sin, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, arctan, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, arccos, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, arcsin, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // HYPERBOLIC
   PHP_ME(CArray, sinh, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, tanh, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, cosh, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // INITIALIZERS SECTION
   PHP_ME(CArray, identity, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, zeros, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, zeros_like, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, ones, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, ones_like, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, eye, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, full, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, full_like, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // BASIC OPERATIONS
   PHP_ME(CArray, sum, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)

   // EXPONENTIAL
   PHP_ME(CArray, exp, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // CONVERT SECTION
   PHP_ME(CArray, toArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, fromArray, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, toDouble, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, fromDouble, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   
   // VISUALIZATION
   PHP_ME(CArray, print_r, NULL, ZEND_ACC_STATIC | ZEND_ACC_PUBLIC)
   PHP_ME(CArray, __toString, NULL, ZEND_ACC_PUBLIC)
   
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
    init_exception_objects();
    INIT_CLASS_ENTRY(ce, "CArray", phpsci_class_methods);
    phpsci_sc_entry = zend_register_internal_class(&ce);
    zend_class_implements(phpsci_sc_entry, 1, zend_ce_arrayaccess);
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
