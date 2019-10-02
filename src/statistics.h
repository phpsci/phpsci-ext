#ifndef CARRAY_STATISTICS_H
#define CARRAY_STATISTICS_H

#include "carray.h"
#include "php.h"
#include <Python.h>
#include <numpy/arrayobject.h>

CArray * CArray_Correlate(CArray *a, CArray *b, int mode);
CArray * CArray_Mean(CArray *a, int axis, int dtype);

#endif //CARRAY_STATISTICS_H
