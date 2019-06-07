#ifndef PHPSCI_EXT_TRIGONOMETRIC_H
#define PHPSCI_EXT_TRIGONOMETRIC_H

#include "carray.h"

typedef double (CArray_CFunc_ElementWise) (double x);

CArray * CArray_Sin(CArray * target, MemoryPointer * out);
#endif //PHPSCI_EXT_TRIGONOMETRIC_H
