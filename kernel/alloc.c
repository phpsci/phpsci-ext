//
// Created by Henrique Borba on 19/11/2018.
//

#include "alloc.h"
#include "carray.h"

/**
 * Alocates CArray Data Buffer based on numElements and elsize from 
 * CArray descriptor.
 **/ 
void
CArray_Data_alloc(CArray * ca)
{
    ca->data = (void*)emalloc((ca->descriptor->numElements * ca->descriptor->elsize));
}




