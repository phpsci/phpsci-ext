#include "carray.h"
#include "php.h"
#include "buffer.h"
#include <Python.h>
#include <numpy/arrayobject.h>


static void
Hashtable_copydata(zval * target_zval, CArray * target_carray, int * first_index)
{
    zval * element;
    long * data_long;
    double * data_double;

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
                ZVAL_DEREF(element);
                if (Z_TYPE_P(element) == IS_ARRAY) {
                    Hashtable_copydata(element, target_carray, first_index);
                }
                if (Z_TYPE_P(element) == IS_LONG) {
                    if (PyArray_TYPE(target_carray) == NPY_LONG) {
                        convert_to_long(element);
                        data_long = (long *) PyArray_DATA(target_carray);
                        data_long[*first_index] = zval_get_long(element);
                        *first_index = *first_index + 1;
                    }
                    if (PyArray_TYPE(target_carray) == NPY_DOUBLE) {
                        convert_to_long(element);
                        data_double = (double *) PyArray_DATA(target_carray);
                        data_double[*first_index] = (double) zval_get_long(element);
                        *first_index = *first_index + 1;
                    }
                }
                if (Z_TYPE_P(element) == IS_DOUBLE) {
                    if (PyArray_TYPE(target_carray) == NPY_DOUBLE) {
                        convert_to_double(element);
                        data_double = (double *) PyArray_DATA(target_carray);
                        data_double[*first_index] = (double) zval_get_double(element);
                        *first_index = *first_index + 1;
                    }
                    if (PyArray_TYPE(target_carray) == NPY_LONG) {
                        convert_to_double(element);
                        data_long = (long *) PyArray_DATA(target_carray);
                        data_long[*first_index] = (long) zval_get_double(element);
                        *first_index = *first_index + 1;
                    }
                }
            } ZEND_HASH_FOREACH_END();
}

static void
Hashtable_type(zval * target_zval, int * type)
{
    zval * element;

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
                ZVAL_DEREF(element);
                if (Z_TYPE_P(element) == IS_ARRAY) {
                    Hashtable_type(element, type);
                    break;
                }
                if (Z_TYPE_P(element) == IS_LONG) {
                    *type = NPY_LONG;
                    break;
                }
                if (Z_TYPE_P(element) == IS_DOUBLE) {
                    *type = NPY_DOUBLE;
                    break;
                }
            } ZEND_HASH_FOREACH_END();
}

static void
Hashtable_dimensions(zval * target_zval, npy_intp * dims, int ndims, int current_dim)
{
    zval * element;
    if (Z_TYPE_P(target_zval) == IS_ARRAY) {
        dims[current_dim] = (npy_intp)zend_hash_num_elements(Z_ARRVAL_P(target_zval));
    }

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
                ZVAL_DEREF(element);
                if (Z_TYPE_P(element) == IS_ARRAY) {
                    Hashtable_dimensions(element, dims, ndims, (current_dim + 1));
                    break;
                }
            } ZEND_HASH_FOREACH_END();
}

static void
Hashtable_ndim(zval * target_zval, int * ndims)
{
    zval * element;
    int current_dim = *ndims;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
                ZVAL_DEREF(element);
                if (Z_TYPE_P(element) == IS_ARRAY) {
                    *ndims = *ndims + 1;
                    Hashtable_ndim(element, ndims);
                    break;
                }
            } ZEND_HASH_FOREACH_END();
}


CArray *
CArray_NewFromArrayPHP(zval *arr, int type)
{
    PyObject *sc;
    PyArray_Descr *dtype;
    CArray *new_carray;
    int ndims = 1;
    int last_index = 0;
    npy_intp *dims;

    if (type == INT_MAX) {
        type = NPY_DOUBLE;
    }

    if (PyArray_API == NULL) {
        import_array();
    }

    if (Z_TYPE_P(arr) == IS_LONG) {
        convert_to_long(arr);
        return (CArray *)((PyArrayObject *)PyLong_FromLong(zval_get_long(arr)));
    }
    if (Z_TYPE_P(arr) == IS_DOUBLE) {
        convert_to_double(arr);
        return (CArray *)((PyArrayObject *)PyFloat_FromDouble(zval_get_long(arr)));
    }

    Hashtable_ndim(arr, &ndims);
    dims = (npy_intp*)emalloc(ndims * sizeof(npy_intp));
    Hashtable_dimensions(arr, dims, ndims, 0);

    if (type == INT_MAX) {
        Hashtable_type(arr, &type);
    }

    new_carray = (CArray *)PyArray_SimpleNew(ndims, dims, type);
    Hashtable_copydata(arr, new_carray, &last_index);

    efree(dims);
    return new_carray;
}

void
CArray_Dump(CArray *ca)
{
    int i;
    php_printf("CArray.dims\t\t\t[");
    for(i = 0; i < PyArray_NDIM(ca); i ++) {
        php_printf(" %d", (int)PyArray_DIM(ca, i));
    }
    php_printf(" ]\n");
    php_printf("CArray.strides\t\t\t[");
    for(i = 0; i < PyArray_NDIM(ca); i ++) {
        php_printf(" %d", (int)PyArray_STRIDE(ca, i));
    }
    php_printf(" ]\n");
    php_printf("CArray.ndim\t\t\t%d\n", PyArray_NDIM(ca));
    php_printf("CArray.flags\t\t\t");
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_C_CONTIGUOUS)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_C_CONTIGUOUS ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_F_CONTIGUOUS)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_F_CONTIGUOUS ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_ALIGNED)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_ALIGNED ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_WRITEABLE)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_WRITEABLE ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_OWNDATA)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_OWNDATA ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_UPDATE_ALL)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_UPDATE_ALL ");
    }
    if(PyArray_CHKFLAGS(ca, NPY_ARRAY_UPDATEIFCOPY)) {
        php_printf("\n\t\t\t\tCARRAY_ARRAY_UPDATEIFCOPY ");
    }
    php_printf("\n");
    php_printf("CArray.descriptor.elsize\t%d\n", PyArray_ITEMSIZE(ca));
    php_printf("CArray.descriptor.numElements\t%d\n", (int)PyArray_SIZE(ca));
    php_printf("CArray.descriptor.type\t\t%d\n", PyArray_TYPE(ca));
}

CArray *
CArray_Print(CArray *a)
{
    if (PyArray_API == NULL) {
        import_array();
    }

    PyObject* objectRepresentation = PyObject_Repr((PyObject *) a);
    php_printf(PyBytes_AS_STRING(PyUnicode_AsEncodedString(objectRepresentation, "utf-8",
                                         "Error ~")));
    php_printf("\n");
}

CArray *
CArray_FromMemoryPointer(MemoryPointer *ptr)
{
    return get_from_buffer(ptr->uuid);
}

CArray *
CArray_Ones(CArray *shape, int typenum)
{
    PyObject *scalar = Py_BuildValue("i", 1);
    CArray *newcarray;

    newcarray = (CArray *)PyArray_SimpleNew(PyArray_SIZE((CArray *)shape), (npy_intp*)PyArray_DATA(shape), typenum);
    PyArray_FillWithScalar((PyArrayObject *)newcarray, scalar);
    return newcarray;
}

CArray *
CArray_Zeros(CArray *shape, int typenum)
{
    PyObject *scalar = Py_BuildValue("i", 0);
    CArray *newcarray;

    CArray_Print(shape);

    newcarray = (CArray *)PyArray_SimpleNew(PyArray_SIZE((CArray *)shape), (npy_intp*)PyArray_DATA(shape), NPY_DOUBLE);
    if(PyArray_FillWithScalar((PyArrayObject *)newcarray, scalar) < 0) {
        php_printf("ERROR");
    }

    return newcarray;
}

CArray *
CArray_Arange(double start, double stop, double step, int typenum)
{
    if (PyArray_API == NULL) {
        import_array();
    }
    CArray *rtn = (CArray *)PyArray_Arange(start, stop, step, typenum);

    return rtn;
}