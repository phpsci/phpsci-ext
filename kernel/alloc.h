//
// Created by Henrique Borba on 19/11/2018.
//

#ifndef PHPSCI_EXT_ALLOC_H
#define PHPSCI_EXT_ALLOC_H

#include "carray.h"

void CArray_Data_alloc(CArray * ca);
void * carray_data_alloc_zeros(int size);
void * carray_data_alloc(uintptr_t size);

void CArray_INCREF(CArray * target);
void CArray_DECREF(CArray * target);
#endif //PHPSCI_EXT_ALLOC_H
