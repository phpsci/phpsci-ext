#ifndef CARRAY_SEARCH_H
#define CARRAY_SEARCH_H

#include "carray.h"

int INT_argmax(int *ip, int n, int *max_ind, CArray *CARRAY_UNUSED(aip));
int DOUBLE_argmax(double *ip, int n, int *max_ind, CArray *CARRAY_UNUSED(aip));

CArray * CArray_Argmax(CArray * target, int * axis, MemoryPointer * out);

#endif //CARRAY_SEARCH_H
