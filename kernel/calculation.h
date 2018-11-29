#ifndef PHPSCI_EXT_CALCULATION_H
#define PHPSCI_EXT_CALCULATION_H

#include "carray.h"

CArray * CArray_Sum(CArray * self, int * axis, int rtype, MemoryPointer * out_ptr);
CArray * CArray_Prod(CArray * self, int * axis, int rtype, MemoryPointer * out_ptr);
#endif 