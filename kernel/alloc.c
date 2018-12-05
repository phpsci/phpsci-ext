//
// Created by Henrique Borba on 19/11/2018.
//

#include "alloc.h"
#include "carray.h"
#include "buffer.h"

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
carray_data_alloc_zeros(int num_elements, int size_element, char type)
{
    int i;
    void * data;
    data = ecalloc(num_elements, size_element);

    if(type == TYPE_INTEGER) {
        for(i = 0; i < num_elements; i++) {
            ((int*)data)[i] = 0;
        }
    }
    if(type == TYPE_DOUBLE) {
        for(i = 0; i < num_elements; i++) {
            ((double*)data)[i] = 0.00;
        }
    }
    return (void *)data;
}

/**
 * @param size
 * @return
 */
void *
carray_data_alloc(int num_elements, int size_element)
{
    return (void*)emalloc(num_elements * size_element);
}

/**
 * @param descriptor
 */
void
CArrayDescriptor_INCREF(CArrayDescriptor * descriptor)
{
    descriptor->refcount++;
}

/**
 * @param descriptor
 */
void
CArrayDescriptor_DECREF(CArrayDescriptor * descriptor)
{
    descriptor->refcount--;
}

/**
 * Free CArrays owning data buffer
 */  
void
_free_data_owner(MemoryPointer * ptr)
{
    CArray * array = CArray_FromMemoryPointer(ptr);
    if(array->descriptor->refcount == 0) {
        efree(array->descriptor);
    }
    if(array->refcount == 0) {
        efree(array->data);
    }
}

/**
 * Free CArrays that refers others CArrays
 */  
void
_free_data_ref(MemoryPointer * ptr)
{
    CArray * array = CArray_FromMemoryPointer(ptr);
    
    if(array->refcount == 0 && array->base->refcount <= 1) {
        efree(array->data);
    }
    if(array->refcount == 0 && array->base->refcount > 1) {
        CArray_DECREF(array->base);
    }
}

/**
 * Free CArray using MemoryPointer
 **/ 
void
CArray_Alloc_FreeFromMemoryPointer(MemoryPointer * ptr)
{
    CArray * array = CArray_FromMemoryPointer(ptr);
    if(CArray_CHKFLAGS(array, CARRAY_ARRAY_OWNDATA)){
        _free_data_owner(ptr);
        return;
    }
    _free_data_ref(ptr);
}
