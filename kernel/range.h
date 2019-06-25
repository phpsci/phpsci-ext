#ifndef CARRAY_RANGE_H
#define CARRAY_RANGE_H

#include "carray.h"

CArray * CArray_Linspace(double start, double stop, int num, int endpoint, int retstep, int axis, int type, MemoryPointer * out);
CArray * CArray_Arange(double start, double stop, double step, int type_num, MemoryPointer * ptr);

#endif //CARRAY_RANGE_H
