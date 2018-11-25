//
// Created by Henrique Borba on 25/11/2018.
//

#ifndef PHPSCI_EXT_SHAPE_H
#define PHPSCI_EXT_SHAPE_H

#include "carray.h"

CArray * CArray_Newshape(CArray * self, int *newdims, int new_ndim, CARRAY_ORDER order, MemoryPointer * ptr);

#endif //PHPSCI_EXT_SHAPE_H
