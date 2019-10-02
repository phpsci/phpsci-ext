#ifndef CARRAY_LINALG_H
#define CARRAY_LINALG_H

#include <Python.h>
#include "carray.h"
#include <numpy/arrayobject.h>

CArray *
CArray_Einsum(char* subscripts, int nop, CArray ** op_in, int typenum,
              NPY_ORDER order, NPY_CASTING casting);

#endif //CARRAY_LINALG_H
