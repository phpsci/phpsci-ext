//
// Created by Henrique Borba on 25/11/2018.
//

#ifndef PHPSCI_EXT_SHAPE_H
#define PHPSCI_EXT_SHAPE_H

#include "carray.h"

CArray * CArray_Newshape(CArray * self, int *newdims, int new_ndim, CARRAY_ORDER order, MemoryPointer * ptr);
CArray * CArray_Transpose(CArray * target, CArray_Dims * permute, MemoryPointer * ptr);
CArray * CArray_SwapAxes(CArray * ap, int a1, int a2, MemoryPointer * out);
void     CArray_CreateSortedStridePerm(int ndim, int *strides, ca_stride_sort_item *out_strideperm);
CArray * CArray_Ravel(CArray *arr, CARRAY_ORDER order);
#endif //PHPSCI_EXT_SHAPE_H
