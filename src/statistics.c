#include "carray.h"
#include "php.h"
#include "statistics.h"
#include "exceptions.h"
#include <Python.h>
#include <numpy/arrayobject.h>


CArray *
CArray_Correlate(CArray *a, CArray *b, int mode)
{
    CArray * rtn;
    if (mode == INT_MAX) {
        mode = 0;
    }

    if (PyArray_API == NULL) {
        import_array();
    }

    rtn = (CArray*)PyArray_Correlate2((PyObject*)a, (PyObject*)b, mode);

    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}

CArray *
CArray_Mean(CArray *a, int axis, int dtype)
{
    CArray *rtn;
    if (PyArray_API == NULL) {
        import_array();
    }
    rtn = (CArray *)PyArray_Mean((PyArrayObject*)a, axis, dtype, NULL);
    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}