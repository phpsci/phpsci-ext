#include "numeric.h"
#include "carray.h"
#include "exceptions.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

CArray *
CArray_GenericUnaryFunction(char * op, CArray *a)
{
    if (PyArray_API == NULL) {
        import_array();
    }

    PyObject *n_ops = PyArray_GetNumericOps();
    PyArray_SetNumericOps(n_ops);

    CArray *rtn = (CArray *)PyObject_CallFunctionObjArgs(PyDict_GetItemString(n_ops, op), (PyObject*)a, NULL);

    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}

CArray *
CArray_GenericBinaryFunction(char * op, CArray *a, CArray *b)
{
    if (PyArray_API == NULL) {
        import_array();
    }
    PyObject *n_ops = PyArray_GetNumericOps();
    PyArray_SetNumericOps(n_ops);
    CArray *rtn = (CArray *)PyObject_CallFunctionObjArgs(PyDict_GetItemString(n_ops, op), (PyObject*)a, (PyObject*)b, NULL);

    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}

CArray *
CArray_Add(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("add", a, b);
}

CArray *
CArray_Subtract(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("subtract", a, b);
}

CArray *
CArray_Multiply(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("multiply", a, b);
}

CArray *
CArray_Divide(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("divide", a, b);
}

CArray *
CArray_Remainder(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("remainder", a, b);
}

CArray *
CArray_Divmod(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("divmod", a, b);
}

CArray *
CArray_Power(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("power", a, b);
}

CArray *
CArray_LeftShift(CArray *a, CArray *b)
{
    return CArray_GenericBinaryFunction("left_shift", a, b);
}


/**
 * Unary Functions
 */
CArray *
CArray_Positive(CArray *a)
{
    return CArray_GenericUnaryFunction("positive", a);
}

CArray *
CArray_Negative(CArray *a)
{
    return CArray_GenericUnaryFunction("negative", a);
}

CArray *
CArray_Absolute(CArray *a)
{
    return CArray_GenericUnaryFunction("absolute", a);
}

CArray *
CArray_Invert(CArray *a)
{
    return CArray_GenericUnaryFunction("invert", a);
}

CArray *
CArray_Ceil(CArray *a)
{
    return CArray_GenericUnaryFunction("ceil", a);
}

CArray *
CArray_Floor(CArray *a)
{
    return CArray_GenericUnaryFunction("floor", a);
}

CArray *
CArray_Sum(CArray *a, int axis, int dtype)
{
    CArray *rtn;
    if (PyArray_API == NULL) {
        import_array();
    }
    rtn = (CArray *)PyArray_Sum((PyArrayObject*)a, axis, dtype, NULL);
    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}

CArray *
CArray_Prod(CArray *a, int axis, int dtype)
{
    CArray *rtn;
    if (PyArray_API == NULL) {
        import_array();
    }
    rtn = (CArray *)PyArray_Prod((PyArrayObject*)a, axis, dtype, NULL);
    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}