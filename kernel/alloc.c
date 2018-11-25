//
// Created by Henrique Borba on 19/11/2018.
//

#include "alloc.h"
#include "carray.h"

/**
 * @return
 */
void
CArray_INCREF(CArray *target)
{
    target->refcount++;
}

/**
 * @return
 */
void
CArray_DECREF(CArray *target)
{
    target->refcount--;
}

/**
 * Alocates CArray Data Buffer based on numElements and elsize from 
 * CArray descriptor.
 **/ 
void
CArray_Data_alloc(CArray * ca)
{
    ca->data = emalloc((ca->descriptor->numElements * ca->descriptor->elsize));
}

/**
 * @param size
 * @return
 */
void *
carray_data_alloc_zeros(int size)
{
    int i;
    int * data;
    data = (int *)emalloc(size);
    for(i = 0; i < size; i++) {
        data[i] = 0;
    }
    return (void *)data;
}

/**
 * @param size
 * @return
 */
void *
carray_data_alloc(uintptr_t size)
{
    return (void*)emalloc(size);
}


