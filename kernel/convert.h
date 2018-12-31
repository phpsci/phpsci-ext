//
// Created by Henrique Borba on 25/11/2018.
//

#ifndef PHPSCI_EXT_CONVERT_H
#define PHPSCI_EXT_CONVERT_H

#include "carray.h"

CArray * CArray_Slice_Index(CArray * self, int index, MemoryPointer * out);
CArray * CArray_View(CArray *self);
CArray * CArray_NewCopy(CArray *obj, CARRAY_ORDER order);
int CArray_CanCastTo(CArrayDescriptor *from, CArrayDescriptor *to);
int CArray_CanCastSafely(int fromtype, int totype);
int CArray_CastTo(CArray *out, CArray *mp);
#endif //PHPSCI_EXT_CONVERT_H
