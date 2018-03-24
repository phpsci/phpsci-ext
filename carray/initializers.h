//
// Created by hborba on 23/03/18.
//

#ifndef PHPSCI_EXT_INITIALIZERS_H
#define PHPSCI_EXT_INITIALIZERS_H
#include "../phpsci.h"
#include "../kernel/carray.h"
void identity(CArray * carray, int xy);


void zeros2d(CArray * carray, int x, int y);
void zeros(CArray * carray, int x, int y);
void zeros1d(CArray * carray, int x);
#endif //PHPSCI_EXT_INITIALIZERS_H
