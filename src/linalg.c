#include "linalg.h"
#include "carray.h"
#include "exceptions.h"
#include <Python.h>
#include <numpy/arrayobject.h>

CArray *
CArray_Einsum(char* subscripts, int nop, CArray ** op_in, int typenum,
              NPY_ORDER order, NPY_CASTING casting)
{
    CArray * rtn;
    PyArray_Descr *dtype;

    if (PyArray_API == NULL) {
        import_array();
    }

    if (typenum == INT_MAX) {
        dtype = NULL;
    }

    rtn = (CArray *)PyArray_EinsteinSum(subscripts, nop, (PyArrayObject **)op_in, NULL, order, casting, NULL);

    if(PyErr_Occurred()) {
        throw_python_exception();
    }

    return rtn;
}