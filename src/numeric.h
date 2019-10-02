#ifndef CARRAY_NUMERIC_H
#define CARRAY_NUMERIC_H

#include "carray.h"
#include <Python.h>

/**
 * Binary Functions
 */
CArray * CArray_Add(CArray *a, CArray *b);
CArray * CArray_Subtract(CArray *a, CArray *b);
CArray * CArray_Multiply(CArray *a, CArray *b);
CArray * CArray_Divide(CArray *a, CArray *b);
CArray * CArray_Remainder(CArray *a, CArray *b);
CArray * CArray_Divmod(CArray *a, CArray *b);
CArray * CArray_Power(CArray *a, CArray *b);
CArray * CArray_LeftShift(CArray *a, CArray *b);

/**
 * Unary Functions
 */
CArray * CArray_Positive(CArray *a);
CArray * CArray_Negative(CArray *a);
CArray * CArray_Absolute(CArray *a);
CArray * CArray_Invert(CArray *a);
CArray * CArray_Ceil(CArray *a);
CArray * CArray_Floor(CArray *a);

/**
 * Misc
 */
CArray * CArray_Sum(CArray *a, int axis, int dtype);
CArray * CArray_Prod(CArray *a, int axis, int dtype);
#endif //CARRAY_NUMERIC_H
