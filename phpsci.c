#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wimplicit-function-declaration"
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
  | distributed under the License is distributed on an "AS \IS" BASIS,    |
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

#include "src/buffer.h"
#include "src/carray.h"
#include "src/numeric.h"
#include "src/exceptions.h"
#include "src/linalg.h"
#include "src/statistics.h"
#include <Python.h>
#include <numpy/arrayobject.h>

PyThreadState *tstate;

typedef struct _zend_carray_cdata {
    zend_object std;
} end_carray_cdata;

static inline CArray*
ZVAL_TO_CARRAY(zval *obj, int *index2)
{
    int index;
    if (Z_TYPE_P(obj) == IS_OBJECT) {
        zval rv;
        index = (int) zval_get_long(zend_read_property(carray_sc_entry, obj, "uuid", sizeof("uuid") - 1, 1, &rv));

        if (index2 != NULL) {
            *index2 = index;
        }

        return get_from_buffer(index);
    }
    if (Z_TYPE_P(obj) == IS_ARRAY || Z_TYPE_P(obj) == IS_LONG || Z_TYPE_P(obj) == IS_DOUBLE) {
        return CArray_NewFromArrayPHP(obj, INT_MAX);
    }
}

static
void RETURN_MEMORYPOINTER(zval * return_value, MemoryPointer * ptr)
{
    object_init_ex(return_value, carray_sc_entry);
    CArray * arr = CArray_FromMemoryPointer(ptr);
    zend_update_property_long(carray_sc_entry, return_value, "uuid", sizeof("uuid") - 1, ptr->uuid);
    zend_update_property_long(carray_sc_entry, return_value, "ndim", sizeof("ndim") - 1, PyArray_NDIM((PyArrayObject *) arr));
}

static inline zend_object *carray_create_object(zend_class_entry *ce) /* {{{ */
{
    end_carray_cdata * intern = emalloc(sizeof(end_carray_cdata) + zend_object_properties_size(ce));

    zend_object_std_init(&intern->std, ce);
    object_properties_init(&intern->std, ce);

    intern->std.handlers = &carray_object_handlers;

    return &intern->std;
}

PyThreadState *
GTS()
{
    return tstate;
}

PHP_METHOD(CArray, __construct)
{
    MemoryPointer ptr;
    zval * obj_zval;
    CArray * rtn;
    long type;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(obj_zval)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(type)
    ZEND_PARSE_PARAMETERS_END();
    zval * obj = getThis();

    if (ZEND_NUM_ARGS() == 1) {
        type = INT_MAX;
    }

    rtn = CArray_NewFromArrayPHP(obj_zval, (int)type);
    add_to_buffer(&ptr, rtn, sizeof(CArray));

    zend_update_property_long(carray_sc_entry, obj, "uuid", sizeof("uuid") - 1, (int)ptr.uuid);
}

/**
 * ZEROS AND ONES
 */
PHP_METHOD(CArray, ones)
{
    MemoryPointer out;
    zval *input;
    long typenum;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(input)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(typenum)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() == 1) {
        typenum = NPY_FLOAT64;
    }

    CArray *shape = ZVAL_TO_CARRAY(input, NULL);
    CArray *rtn = CArray_Ones(shape, (int) typenum);
    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(rtn));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, zeros)
{
    MemoryPointer out;
    zval *input;
    long typenum;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(input)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(typenum)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() == 1) {
        typenum = NPY_FLOAT64;
    }

    CArray *shape = ZVAL_TO_CARRAY(input, NULL);
    CArray *rtn = CArray_Zeros(shape, (int) typenum);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(rtn));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}



/**
 * NUMERICAL RANGES
 */
PHP_METHOD(CArray, arange)
{
    MemoryPointer out;
    double startd, stopd, stepd;
    long typenum;
    zval *start_stop, *stop_start, *step;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_ZVAL(start_stop)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL(stop_start)
        Z_PARAM_ZVAL(step)
        Z_PARAM_LONG(typenum)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() == 1) {

        startd = 0.00;
        stopd = zval_get_double(start_stop);
        stepd = 1.0;
        typenum = NPY_DOUBLE;
    }
    if (ZEND_NUM_ARGS() == 2) {
        convert_to_double(start_stop);
        convert_to_double(stop_start);
        startd = zval_get_double(start_stop);
        stopd = zval_get_double(stop_start);
        stepd = 1.0;
        typenum = NPY_DOUBLE;
    }
    if (ZEND_NUM_ARGS() == 3) {
        convert_to_double(start_stop);
        convert_to_double(stop_start);
        convert_to_double(step);
        startd = zval_get_double(start_stop);
        stopd = zval_get_double(stop_start);
        stepd = zval_get_double(step);
        typenum = NPY_DOUBLE;
    }
    if (ZEND_NUM_ARGS() == 4) {
        startd = zval_get_double(start_stop);
        stopd = zval_get_double(stop_start);
        stepd = zval_get_double(step);
    }

    CArray *rtn = CArray_Arange(startd, stopd, stepd, (int)typenum);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}

/**
 * ARITHMETICS
 */
PHP_METHOD(CArray, add)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(x1)
        Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Add(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, subtract)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Subtract(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, multiply)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Multiply(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, divide)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Divide(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, remainder)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Remainder(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, divmod)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Divmod(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, power)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Power(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, positive)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Positive(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, negative)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Negative(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, absolute)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Absolute(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, invert)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Invert(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, ceil)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Ceil(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, floor)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(x1)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    CArray *rtn = CArray_Floor(x1c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}

PHP_METHOD(CArray, left_shift)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_LeftShift(x1c, x2c);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}

/**
 * SUM, PRODUCTS & DIFFERENCES
 */
PHP_METHOD(CArray, sum)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    long axis = NPY_MAXDIMS, dtype;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_ZVAL(x1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
        Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    if (ZEND_NUM_ARGS() == 1) {
        axis = NPY_MAXDIMS;
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }
    if (ZEND_NUM_ARGS() == 2) {
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }

    CArray *rtn = CArray_Sum(x1c, (int)axis, (int)dtype);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, prod)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    long axis = NPY_MAXDIMS, dtype;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
            Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    if (ZEND_NUM_ARGS() == 1) {
        axis = NPY_MAXDIMS;
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }
    if (ZEND_NUM_ARGS() == 2) {
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }

    CArray *rtn = CArray_Prod(x1c, (int)axis, (int)dtype);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}
PHP_METHOD(CArray, mean)
{
    MemoryPointer out;
    CArray *x1c;
    zval *x1;
    long axis = NPY_MAXDIMS, dtype;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
            Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    x1c = ZVAL_TO_CARRAY(x1, NULL);

    if (ZEND_NUM_ARGS() == 1) {
        axis = NPY_MAXDIMS;
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }
    if (ZEND_NUM_ARGS() == 2) {
        dtype = PyArray_TYPE((PyArrayObject*)x1c);
    }

    CArray *rtn = CArray_Mean(x1c, (int)axis, (int)dtype);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}

/**
 * STATISTICS
 */
PHP_METHOD(CArray, correlate)
{
    MemoryPointer out;
    CArray *x1c, *x2c;
    zval *x1, *x2;
    long mode;
    ZEND_PARSE_PARAMETERS_START(2, 3)
            Z_PARAM_ZVAL(x1)
            Z_PARAM_ZVAL(x2)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(mode)
    ZEND_PARSE_PARAMETERS_END();

    if (ZEND_NUM_ARGS() == 2) {
        mode = INT_MAX;
    }

    x1c = ZVAL_TO_CARRAY(x1, NULL);
    x2c = ZVAL_TO_CARRAY(x2, NULL);

    CArray *rtn = CArray_Correlate(x1c, x2c, (int)mode);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
}

/**
 * LINEAR ALGEBRA
 */
PHP_METHOD(CArray, einsum)
{
    MemoryPointer out;
    int i;
    char * subscripts;
    size_t sub_len;
    CArray ** ops;
    zval * dict;
    int dict_size = 0;
    ZEND_PARSE_PARAMETERS_START(2, -1)
            Z_PARAM_STRING(subscripts, sub_len)
            Z_PARAM_VARIADIC('+', dict, dict_size)
    ZEND_PARSE_PARAMETERS_END();

    ops = (CArray **)emalloc(sizeof(CArray*) * dict_size);
    for (i = 0; i < dict_size; i++) {
        ops[i] = ZVAL_TO_CARRAY(&(dict[i]), NULL);
    }

    CArray *rtn = CArray_Einsum(subscripts, dict_size, ops, INT_MAX, NPY_KEEPORDER, NPY_SAFE_CASTING);

    if (rtn != NULL) {
        add_to_buffer(&out, rtn, sizeof(CArray));
        RETURN_MEMORYPOINTER(return_value, &out);
    }
    efree(ops);
}



/**
 * DESTRUCTOR
 **/
PHP_METHOD(CArray, __destruct)
{
    int uuid;
    CArray * target;
    target = ZVAL_TO_CARRAY(getThis(), &uuid);
    Py_DECREF((PyArrayObject *) target);
    buffer_remove_index(uuid);
}

PHP_METHOD(CArray, __toString)
{
    CArray *self;
    zend_string *str = zend_string_init("", 0, 0);
    int uuid;
    ZVAL_TO_CARRAY(getThis(), &uuid);
    self = get_from_buffer(uuid);

    CArray_Print(self);

    ZVAL_STR(return_value, str);
}

/**
 * CLASS METHODS
 */
static zend_function_entry carray_class_methods[] =
{
        PHP_ME(CArray, __construct, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, __destruct, NULL, ZEND_ACC_PUBLIC)
        PHP_ME(CArray, __toString, NULL, ZEND_ACC_PUBLIC)

        // ZEROS AND ONES
        PHP_ME(CArray, ones, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, zeros, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // NUMERICAL RANGES
        PHP_ME(CArray, arange, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // ARITHMETICS
        PHP_ME(CArray, add, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, subtract, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, multiply, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, divide, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, remainder, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, divmod, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, power, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, positive, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, negative, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, absolute, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, invert, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, left_shift, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, ceil, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, floor, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // TRIGONOMETRIC


        // STATISTICS
        PHP_ME(CArray, correlate, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, mean, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // SUM, PRODUCTS & DIFFERENCES
        PHP_ME(CArray, sum, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        PHP_ME(CArray, prod, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)


        // LINEAR ALGEBRA
        PHP_ME(CArray, einsum, NULL, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        { NULL, NULL, NULL }
};
static zend_function_entry carray_iterator_class_methods[] =
{
        { NULL, NULL, NULL }
};
zend_function_entry carray_functions[] = {
        {NULL, NULL, NULL}
};

zval * ca_read_dimension
(zval *object, zval *offset, int type, zval *rv) {
    php_printf("FOI");
    return rv;
}


/**
 * MINIT
 */
static PHP_MINIT_FUNCTION(carray)
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "CArray", carray_class_methods);
    carray_sc_entry = zend_register_internal_class(&ce);


    carray_sc_entry->create_object = carray_create_object;

    memcpy(&carray_object_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    carray_object_handlers.read_dimension = ca_read_dimension;

    init_exception_objects();

    Py_IgnoreEnvironmentFlag = 0;

    Py_InitializeEx(0);
    PyEval_InitThreads();

    REGISTER_LONG_CONSTANT("CARRAY_UINT8", NPY_UINT8, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_UINT", NPY_UINT, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_INT", NPY_INT, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_INT8", NPY_INT8, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_INT32", NPY_INT, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_LONG", NPY_LONG, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_LONGDOUBLE", NPY_LONGDOUBLE, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_LONGLONG", NPY_LONGLONG, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_FLOAT", NPY_FLOAT, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_FLOAT64", NPY_FLOAT64, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_FLOAT16", NPY_FLOAT16, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_FLOAT32", NPY_FLOAT32, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("CARRAY_DOUBLE", NPY_DOUBLE, CONST_CS|CONST_PERSISTENT);

    //MODES
    REGISTER_LONG_CONSTANT("MODE_FULL",  2, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("MODE_SAME",  1, CONST_CS|CONST_PERSISTENT);
    REGISTER_LONG_CONSTANT("MODE_VALID", 0, CONST_CS|CONST_PERSISTENT);


    REGISTER_DOUBLE_CONSTANT("CARRAY_NAN", NAN, CONST_CS|CONST_PERSISTENT);



    return Py_IsInitialized() ? SUCCESS : FAILURE;
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
    /*
     * Shut down the embedded Python interpreter.  This will destroy all of
     * the sub-interpreters and (ideally) free all of the memory allocated
     * by Python.
     */
    Py_Finalize();

    /*
     * Swap out our temporary state and release our lock.  We're now totally
     * done with the Python system.
     */
    PyThreadState_Swap(NULL);
    PyEval_ReleaseLock();


    return SUCCESS;
}

/**
 * RSHUTDOWN
 */
PHP_RSHUTDOWN_FUNCTION(carray)
{
    throw_python_exception();
    Py_Finalize();
    return SUCCESS;
}

/**
 * PHP_RINIT
 */
static PHP_RINIT_FUNCTION(carray)
{
    return SUCCESS;
}

zend_module_entry carray_module_entry = {
        STANDARD_MODULE_HEADER,
        PHP_CARRAY_EXTNAME,
        carray_functions,				/* Functions */
        PHP_MINIT(carray),				/* MINIT */
        PHP_MSHUTDOWN(carray),			/* MSHUTDOWN */
        PHP_RINIT(carray),						    /* RINIT */
        PHP_RSHUTDOWN(carray),			/* RSHUTDOWN */
        PHP_MINFO(carray),				/* MINFO */
        PHP_CARRAY_VERSION,				/* version */
        STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_CARRAY
ZEND_GET_MODULE(carray)
#endif /* COMPILE_DL_CARRAY */

#pragma clang diagnostic pop