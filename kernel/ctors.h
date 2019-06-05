#ifndef CARRAY_CTORS_H
#define CARRAY_CTORS_H

#include "carray.h"

int setArrayFromSequence(CArray *a, CArray *s, int dim, int offset);
CArray * CArray_FromArray(CArray *arr, CArrayDescriptor *newtype, int flags);

#endif //CARRAY_CTORS_H
