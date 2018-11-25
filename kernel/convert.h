//
// Created by Henrique Borba on 25/11/2018.
//

#ifndef PHPSCI_EXT_CONVERT_H
#define PHPSCI_EXT_CONVERT_H

#include "carray.h"

CArray * CArray_View(CArray *self);
CArray * CArray_NewCopy(CArray *obj, CARRAY_ORDER order);

#endif //PHPSCI_EXT_CONVERT_H
