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
#include "php.h"
#include "ext/standard/info.h"
#include "Zend/zend_interfaces.h"

#include "kernel/carray.h"
#include "kernel/iterators.h"
#include "kernel/shape.h"
#include "kernel/calculation.h"
#include "kernel/convert.h"
#include "kernel/common/exceptions.h"
#include "kernel/linalg.h"
#include "kernel/alloc.h"
#include "kernel/number.h"
#include "kernel/trigonometric.h"
#include "kernel/common/exceptions.h"
#include "kernel/shape.h"
#include "kernel/item_selection.h"
#include "kernel/scalar.h"
#include "kernel/random.h"

void ZVAL_TO_MEMORYPOINTER(zval * obj, MemoryPointer * ptr)
{
    zval rv;
    ptr->uuid = (int)zval_get_long(zend_read_property(carray_sc_entry, obj, "uuid", sizeof("uuid") - 1, 1, &rv));
}

void * FREE_TUPLE(int * tuple)
{
    if(tuple != NULL)
        efree(tuple);
}

int * ZVAL_TO_TUPLE(zval * obj, int * size)
{
    zval * element;
    *size = 0;
    int * data_int;
    data_int = (int *)emalloc(zend_hash_num_elements(Z_ARRVAL_P(obj)) * sizeof(int));

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(obj), element) {
        convert_to_long(element);
        data_int[*size] = (int)zval_get_long(element);
        *size = *size + 1;
    } ZEND_HASH_FOREACH_END();
    return data_int;
}

void RETURN_MEMORYPOINTER(zval * return_value, MemoryPointer * ptr)
{
    object_init_ex(return_value, carray_sc_entry);
    CArray * arr = CArray_FromMemoryPointer(ptr);
    zend_update_property_long(carray_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr->uuid);
    zend_update_property_long(carray_sc_entry, return_value, "ndim", sizeof("ndim") - 1, arr->ndim);
}

PHP_METHOD(CArray, __construct)
{
    MemoryPointer ptr;
    zval * obj_zval;
    char * type;
    size_t type_name_len;
    char   type_parsed;

    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(obj_zval)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(type, type_name_len)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        type_parsed = 'a';
    }
    if(ZEND_NUM_ARGS() > 1) {
        type_parsed = type[0];
    }
    CArray_FromZval(obj_zval, type_parsed, &ptr);
    zval * obj = getThis();
    CArray * arr = CArray_FromMemoryPointer(&ptr);
    zend_update_property_long(carray_sc_entry, obj, "uuid", sizeof("uuid") - 1, (int)ptr.uuid);
    zend_update_property_long(carray_sc_entry, obj, "ndim", sizeof("ndim") - 1, (int)arr->ndim);
}

/**
 * GET & SETS
 **/ 
ZEND_BEGIN_ARG_INFO_EX(arginfo_array_set, 0, 0, 2)
    ZEND_ARG_INFO(0, name)
    ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO()
PHP_METHOD(CArray, __set)
{
    size_t name_len;
    char * name;
    zval * value;
    MemoryPointer value_ptr, target_ptr;

    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_STRING(name, name_len)
        Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();
    if(!strcmp(name, "flat")) {
        CArray * value_arr, * self_arr;
        ZVAL_TO_MEMORYPOINTER(getThis(), &target_ptr);
        ZVAL_TO_MEMORYPOINTER(value, &value_ptr);
        value_arr = CArray_FromMemoryPointer(&value_ptr);
        self_arr = CArray_FromMemoryPointer(&target_ptr);
        array_flat_set(self_arr, value_arr);
        return;
    }
    if(!strcmp(name, "uuid")) {
        zend_update_property_long(carray_sc_entry, getThis(), "uuid", sizeof("uuid") - 1, zval_get_long(value));
        return;
    }
    if(!strcmp(name, "ndim")) {
        zend_update_property_long(carray_sc_entry, getThis(), "ndim", sizeof("ndim") - 1, zval_get_long(value));
        return;
    }
    throw_valueerror_exception("Unknown property.");    
}

PHP_METHOD(CArray, offsetExists)
{
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
}

ZEND_BEGIN_ARG_INFO_EX(arginfo_array_offsetGet, 0, 0, 1)
    ZEND_ARG_INFO(0, index)
ZEND_END_ARG_INFO()
PHP_METHOD(CArray, offsetGet)
{
    CArray * _this_ca, * ret_ca;
    MemoryPointer ptr, target_ptr;
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
    convert_to_long(index);
    zval * obj = getThis();
    ZVAL_TO_MEMORYPOINTER(obj, &ptr);
    _this_ca = CArray_FromMemoryPointer(&ptr);
    ret_ca = (CArray *) CArray_Slice_Index(_this_ca, (int)zval_get_long(index), &target_ptr);
    RETURN_MEMORYPOINTER(return_value, &target_ptr);
}
ZEND_BEGIN_ARG_INFO_EX(arginfo_array_offsetSet, 0, 0, 2)
    ZEND_ARG_INFO(0, index)
    ZEND_ARG_INFO(0, newval)
ZEND_END_ARG_INFO()
PHP_METHOD(CArray, offsetSet)
{
}
PHP_METHOD(CArray, offsetUnset)
{
    zval *index;
    if (zend_parse_parameters(ZEND_NUM_ARGS(), "z", &index) == FAILURE) {
        return;
    }
}


PHP_METHOD(CArray, shape)
{
    MemoryPointer ptr;
    CArray * carray, * newcarray;
    zval * new_shape_zval;
    int * new_shape;
    int ndim;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(new_shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    ZVAL_TO_MEMORYPOINTER(getThis(), &ptr);
    carray = CArray_FromMemoryPointer(&ptr);
    new_shape = ZVAL_TO_TUPLE(new_shape_zval, &ndim);
    newcarray = CArray_Newshape(carray, new_shape, zend_hash_num_elements(Z_ARRVAL_P(new_shape_zval)), CARRAY_CORDER, &ptr);
    FREE_TUPLE(new_shape);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}
PHP_METHOD(CArray, dump)
{
    MemoryPointer ptr;
    CArray * array;
    zval * obj = getThis();
    ZVAL_TO_MEMORYPOINTER(obj, &ptr);
    array = CArray_FromMemoryPointer(&ptr);
    CArray_Dump(array);
}

PHP_METHOD(CArray, print)
{
    zval * obj = getThis();
    CArray * array;
    MemoryPointer ptr;
    ZVAL_TO_MEMORYPOINTER(obj, &ptr);
    array = CArray_FromMemoryPointer(&ptr);
    CArray_Print(array);
}

/**
 * DESTRUCTOR
 **/ 
PHP_METHOD(CArray, __destruct)
{
    MemoryPointer ptr;
    ZVAL_TO_MEMORYPOINTER(getThis(), &ptr);
    CArray_Alloc_FreeFromMemoryPointer(&ptr);
}

/**
 * CALCULATIONS
 **/ 
PHP_METHOD(CArray, sum)
{
    zval * target;
    long axis;
    int * axis_p;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        axis_p = NULL;
    }
    if(ZEND_NUM_ARGS() > 1) {
        axis_p = (int*)emalloc(sizeof(int));
        *axis_p = axis;
    }
    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    ret = CArray_Sum(target_ca, axis_p, target_ca->descriptor->type_num, &ptr);
    efree(axis_p);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, sin)
{
    zval * target;
    long axis;
    int * axis_p;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(target)
    ZEND_PARSE_PARAMETERS_END();
    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    ret = CArray_Sin(target_ca, &ptr);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}


PHP_METHOD(CArray, prod)
{
    zval * target;
    long axis;
    int * axis_p;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        axis_p = NULL;
    }
    if(ZEND_NUM_ARGS() > 1) {
        axis_p = (int*)emalloc(sizeof(int));
        *axis_p = axis;
    }
    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    ret = CArray_Prod(target_ca, axis_p, target_ca->descriptor->type_num, &ptr);
    efree(axis_p);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, cumprod)
{
    zval * target;
    long axis;
    int * axis_p;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        axis_p = NULL;
    }
    if(ZEND_NUM_ARGS() > 1) {
        axis_p = (int*)emalloc(sizeof(int));
        *axis_p = axis;
    }
    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    ret = CArray_CumProd(target_ca, axis_p, target_ca->descriptor->type_num, &ptr);
    efree(axis_p);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, cumsum)
{
    zval * target;
    long axis;
    int * axis_p;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        axis_p = NULL;
    }
    if(ZEND_NUM_ARGS() > 1) {
        axis_p = (int*)emalloc(sizeof(int));
        *axis_p = axis;
    }
    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    ret = CArray_CumSum(target_ca, axis_p, target_ca->descriptor->type_num, &ptr);
    efree(axis_p);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, transpose)
{
    zval * target;
    zval * axes;
    int size_axes;
    CArray * ret, * target_ca;
    MemoryPointer ptr;
    CArray_Dims permute;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(target)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY(axes)
    ZEND_PARSE_PARAMETERS_END();

    ZVAL_TO_MEMORYPOINTER(target, &ptr);
    target_ca = CArray_FromMemoryPointer(&ptr);
    if(ZEND_NUM_ARGS() == 1) {
        ret = CArray_Transpose(target_ca, NULL, &ptr);
    }
    if(ZEND_NUM_ARGS() > 1) {
        permute.ptr = ZVAL_TO_TUPLE(axes, &size_axes);
        permute.len = size_axes;
        ret = CArray_Transpose(target_ca, &permute, &ptr);
        FREE_TUPLE(permute.ptr);
    }
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, identity)
{
    MemoryPointer ptr;
    CArray * output;
    zend_long size;
    char * dtype;
    size_t type_len;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_LONG(size)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dtype, type_len)
    ZEND_PARSE_PARAMETERS_END();

    if(ZEND_NUM_ARGS() == 1) {
        dtype = NULL;
    }

    output = CArray_Identity((int)size, dtype, &ptr);
    RETURN_MEMORYPOINTER(return_value, &ptr);
}

PHP_METHOD(CArray, matmul)
{
    MemoryPointer target1_ptr, target2_ptr, result_ptr;

    zval * target1, * target2;
    CArray * target_ca1, * target_ca2, * output_ca, * out;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(target1)
        Z_PARAM_ZVAL(target2)
    ZEND_PARSE_PARAMETERS_END();    
    ZVAL_TO_MEMORYPOINTER(target1, &target1_ptr);
    ZVAL_TO_MEMORYPOINTER(target2, &target2_ptr);
    target_ca1 = CArray_FromMemoryPointer(&target1_ptr);
    target_ca2 = CArray_FromMemoryPointer(&target2_ptr);
    output_ca = CArray_Matmul(target_ca1, target_ca2, NULL, &result_ptr);

    RETURN_MEMORYPOINTER(return_value, &result_ptr);
}
PHP_METHOD(CArray, zeros)
{   
    zval * zshape;
    char * dtype, order = 'C';
    int ndim;
    int * shape;
    MemoryPointer ptr;
    size_t type_len;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(zshape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dtype, type_len)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        dtype = emalloc(sizeof(char));
        *dtype = 'd';
    }
    shape = ZVAL_TO_TUPLE(zshape, &ndim);
    CArray_Zeros(shape, ndim, *dtype, &order, &ptr);
    efree(shape);
    if(ZEND_NUM_ARGS() == 1) {
        efree(dtype);
    }
    RETURN_MEMORYPOINTER(return_value, &ptr);
}
PHP_METHOD(CArray, ones)
{   
    zval * zshape;
    char * dtype, order = 'C';
    int ndim;
    int * shape;
    MemoryPointer ptr;
    size_t type_len;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(zshape)
        Z_PARAM_OPTIONAL
        Z_PARAM_STRING(dtype, type_len)
    ZEND_PARSE_PARAMETERS_END();
    if(ZEND_NUM_ARGS() == 1) {
        dtype = emalloc(sizeof(char));
        *dtype = 'd';
    }
    shape = ZVAL_TO_TUPLE(zshape, &ndim);
    CArray_Ones(shape, ndim, dtype, &order, &ptr);
    efree(shape);
    if(ZEND_NUM_ARGS() == 1) {
        efree(dtype);
    }
    RETURN_MEMORYPOINTER(return_value, &ptr);
}
PHP_METHOD(CArray, add)
{
    MemoryPointer target1_ptr, target2_ptr, result_ptr;
    zval * target1, * target2;
    CArray * target_ca1, * target_ca2, * output_ca, * out;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(target1)
        Z_PARAM_ZVAL(target2)
    ZEND_PARSE_PARAMETERS_END();
    ZVAL_TO_MEMORYPOINTER(target1, &target1_ptr);
    ZVAL_TO_MEMORYPOINTER(target2, &target2_ptr);
    target_ca1 = CArray_FromMemoryPointer(&target1_ptr);
    target_ca2 = CArray_FromMemoryPointer(&target2_ptr);
    output_ca = CArray_Add(target_ca1, target_ca2, &result_ptr);

    RETURN_MEMORYPOINTER(return_value, &result_ptr);
}

/**
 * INDEXING ROUTINES
 */
PHP_METHOD(CArray, diagonal)
{
    MemoryPointer a_ptr, rtn_ptr;
    CArray * target_array;
    zval * a;
    long axis1, axis2, offset;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(offset)
        Z_PARAM_LONG(axis1)
        Z_PARAM_LONG(axis2)
    ZEND_PARSE_PARAMETERS_END();

    if(ZEND_NUM_ARGS() == 1) {
        offset = 0;
        axis1 = 0;
        axis2 = 1;
    }
    if(ZEND_NUM_ARGS() == 2) {
        axis1 = 0;
        axis2 = 1;
    }
    if(ZEND_NUM_ARGS() == 3) {
        axis2 = 1;
    }
    
    ZVAL_TO_MEMORYPOINTER(a, &a_ptr);
    target_array = CArray_FromMemoryPointer(&a_ptr);
    CArray * rtn_array = CArray_Diagonal(target_array, offset, axis1, axis2, &rtn_ptr);
    if(rtn_array == NULL) {
        throw_axis_exception("");
    }
    RETURN_MEMORYPOINTER(return_value, &rtn_ptr);
}

/**
 * MANIPULATION ROUTINES
 */
PHP_METHOD(CArray, swapaxes)
{
    MemoryPointer a_ptr;
    CArray * target_array;
    zval * a;
    long axis1, axis2;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(a)
        Z_PARAM_LONG(axis1)
        Z_PARAM_LONG(axis2)
    ZEND_PARSE_PARAMETERS_END();
    ZVAL_TO_MEMORYPOINTER(a, &a_ptr);
    target_array = CArray_FromMemoryPointer(&a_ptr);
    CArray_SwapAxes(target_array, (int)axis1, (int)axis2, &a_ptr);
    RETURN_MEMORYPOINTER(return_value, &a_ptr);
}

/**
 * NUMERICAL RANGES
 */
PHP_METHOD(CArray, arange)
{
    MemoryPointer a_ptr;
    CArray * target_array;
    double start, stop, step_d;
    int typenum;
    zval * start_stop, * stop_start, * step;
    char * dtype;
    size_t type_len;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(start_stop)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(stop_start)
        Z_PARAM_ZVAL(step)
        Z_PARAM_STRING(dtype, type_len)
    ZEND_PARSE_PARAMETERS_END();

    if(ZEND_NUM_ARGS() == 1) {
        convert_to_double(start_stop);
        start = (double)0.00;
        stop  = (double)zval_get_double(start_stop);
        typenum = TYPE_DEFAULT_INT;
        step_d = 1.00;
    }
    if(ZEND_NUM_ARGS() == 2) {
        convert_to_double(start_stop);
        convert_to_double(stop_start);
        start = (double)zval_get_double(start_stop);
        stop  = (double)zval_get_double(stop_start);
        typenum = TYPE_DEFAULT_INT;
        step_d = 1.00;
    }
    if(ZEND_NUM_ARGS() == 3) {
        convert_to_double(start_stop);
        convert_to_double(stop_start);
        convert_to_double(step);
        start = (double)zval_get_double(start_stop);
        stop  = (double)zval_get_double(stop_start);
        step_d  = (double)zval_get_double(step);
        typenum = TYPE_DEFAULT_INT;
    }
    if(ZEND_NUM_ARGS() == 4) {
        convert_to_double(start_stop);
        convert_to_double(stop_start);
        convert_to_double(step);
        start = (double)zval_get_double(start_stop);
        stop  = (double)zval_get_double(stop_start);
        step_d  = (double)zval_get_double(step);
        typenum = CHAR_TYPE_INT(dtype[0]);
    }
    target_array = CArray_Arange(start, stop, step_d, typenum , &a_ptr);
    RETURN_MEMORYPOINTER(return_value, &a_ptr);
}

/**
 * RANDOM
 **/ 
PHP_METHOD(CArray, rand)
{
    zval * size;
    int len, *dims;
    MemoryPointer out;
    ZEND_PARSE_PARAMETERS_START(1, 1)
       Z_PARAM_ARRAY(size)
    ZEND_PARSE_PARAMETERS_END();
    dims = ZVAL_TO_TUPLE(size, &len);
    CArray_Rand(dims, len, &out);
    RETURN_MEMORYPOINTER(return_value, &out);
    FREE_TUPLE(dims);
}

/**
 * MISC
 **/ 
PHP_METHOD(CArray, fill)
{
    zval * obj = getThis();
    zval * scalar_obj;
    CArrayScalar * scalar;
    MemoryPointer ptr;
    CArray * target_ca;
    ZVAL_TO_MEMORYPOINTER(obj, &ptr);
    ZEND_PARSE_PARAMETERS_START(1, 1)
       Z_PARAM_ZVAL(scalar_obj)
    ZEND_PARSE_PARAMETERS_END();
    if(Z_TYPE_P(scalar_obj) == IS_LONG) {
        convert_to_long(scalar_obj);
        scalar = CArrayScalar_NewInt((int)zval_get_long(scalar_obj));
    }
    if(Z_TYPE_P(scalar_obj) == IS_DOUBLE) {
        convert_to_double(scalar_obj);
        scalar = CArrayScalar_NewDouble(zval_get_double(scalar_obj));
    }
    target_ca = CArray_FromMemoryPointer(&ptr);
    CArray_FillWithScalar(target_ca, scalar);
    CArrayScalar_FREE(scalar);
}


/**
 * CLASS METHODS
 */
static zend_function_entry carray_class_methods[] =
{
        PHP_ME(CArray, __construct, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, __destruct, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, dump, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, print, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, __set, arginfo_array_set, ZEND_ACC_PUBLIC)

        // RANDOM
        PHP_ME(CArray, rand, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // MISC
        PHP_ME(CArray, fill, NULL, ZEND_ACC_PUBLIC)

        // INDEXING
        PHP_ME(CArray, diagonal, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // INITIALIZERS
        PHP_ME(CArray, zeros, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, ones, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // NUMERICAL RANGES
        PHP_ME(CArray, arange, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        //ARRAY MANIPULATION
        PHP_ME(CArray, swapaxes, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // METHODS
        PHP_ME(CArray, identity, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // SHAPE
        PHP_ME(CArray, transpose, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, shape, NULL, ZEND_ACC_PUBLIC)
        
        // LINEAR ALGEBRA
        PHP_ME(CArray, matmul, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // NUMBERS
        PHP_ME(CArray, add, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // CALCULATION
        PHP_ME(CArray, sum, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, prod, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, cumprod, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, cumsum, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // TRIGONOMETRIC
        PHP_ME(CArray, sin, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // CARRAY ITERATOR
        PHP_ME(CArray, offsetUnset, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, offsetSet, arginfo_array_offsetSet, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, offsetGet, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, offsetExists, arginfo_array_offsetGet, ZEND_ACC_PUBLIC)
        { NULL, NULL, NULL }
};
static zend_function_entry carray_iterator_class_methods[] =
{
        { NULL, NULL, NULL }
};
zend_function_entry carray_functions[] = {
        {NULL, NULL, NULL}
};

/**
 * MINIT
 */
static PHP_MINIT_FUNCTION(carray)
{
    zend_class_entry ce;
    memcpy(&carray_object_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    INIT_CLASS_ENTRY(ce, "CArray", carray_class_methods);
    carray_sc_entry = zend_register_internal_class(&ce);
    zend_class_implements(carray_sc_entry, 1, zend_ce_arrayaccess);
    init_exception_objects();
    return SUCCESS;
}

/**
 * MINFO
 */
static PHP_MINFO_FUNCTION(carray)
{
    php_info_print_table_start();
    php_info_print_table_row(2, "CArray support", "enabled");
    php_info_print_table_row(2, "CArray version", PHP_CARRAY_VERSION);
    php_info_print_table_end();
}

/**
 * MSHUTDOWN
 */
static PHP_MSHUTDOWN_FUNCTION(carray)
{
    UNREGISTER_INI_ENTRIES();
    return SUCCESS;
}

zend_module_entry carray_module_entry = {
        STANDARD_MODULE_HEADER,
        PHP_CARRAY_EXTNAME,
        carray_functions,				/* Functions */
        PHP_MINIT(carray),				/* MINIT */
        PHP_MSHUTDOWN(carray),			/* MSHUTDOWN */
        NULL,						    /* RINIT */
        NULL,						    /* RSHUTDOWN */
        PHP_MINFO(carray),				/* MINFO */
        PHP_CARRAY_VERSION,				/* version */
        STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_CARRAY
ZEND_GET_MODULE(carray)
#endif /* COMPILE_DL_CARRAY */
