#ifndef PHPSCI_EXT_ITEM_SELECTION_H
#define PHPSCI_EXT_ITEM_SELECTION_H

#include "carray.h"

CArray * CArray_Diagonal(CArray *self, int offset, int axis1, int axis2, MemoryPointer * rtn);

#endif //PHPSCI_EXT_ITEM_SELECTION_H