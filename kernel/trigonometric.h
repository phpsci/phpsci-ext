#ifndef PHPSCI_EXT_TRIGONOMETRIC_H
#define PHPSCI_EXT_TRIGONOMETRIC_H

#include "carray.h"

typedef double (CArray_CFunc_ElementWise) (double x);

CArray * CArray_Sin(CArray * target, MemoryPointer * out);
CArray * CArray_Cos(CArray * target, MemoryPointer * out);
CArray * CArray_Tan(CArray * target, MemoryPointer * out);
CArray * CArray_Arcsin(CArray * target, MemoryPointer * out);
CArray * CArray_Arccos(CArray * target, MemoryPointer * out);
CArray * CArray_Arctan(CArray * target, MemoryPointer * out);
CArray * CArray_Sinh(CArray * target, MemoryPointer * out);
CArray * CArray_Cosh(CArray * target, MemoryPointer * out);
CArray * CArray_Tanh(CArray * target, MemoryPointer * out);
CArray * CArray_Arcsinh(CArray * target, MemoryPointer * out);
CArray * CArray_Arccosh(CArray * target, MemoryPointer * out);
CArray * CArray_Arctanh(CArray * target, MemoryPointer * out);
#endif //PHPSCI_EXT_TRIGONOMETRIC_H
