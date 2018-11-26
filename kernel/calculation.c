#include "calculation.h"
#include "carray.h"
#include "buffer.h"

/**
 * CArray Sum
 **/ 
CArray *
CArray_Sum(CArray * self, int * axis, int rtype, MemoryPointer * out_ptr)
{
    int i, j = 0;
    void * total;
    CArray * arr, * ret = NULL;
    CArrayDescriptor * descr;
    ret = (CArray *)emalloc(sizeof(CArray));
    descr = (CArrayDescriptor*)emalloc(sizeof(CArrayDescriptor));
    arr = CArray_CheckAxis(self, axis, 0);
    int index_jumps = self->strides[*axis]/self->descriptor->elsize;
    
    if (arr == NULL) {
        return NULL;
    }

    switch(rtype) {
        case TYPE_INTEGER_INT:
            total = (int *)emalloc(sizeof(int));
            *((int *)total) = 0;
            break;
        case TYPE_DOUBLE_INT:
            total = (double *)emalloc(sizeof(double));
            *((double *)total) = 0.00;
            break;
        default:
            total = (double *)emalloc(sizeof(double));
            *((double *)total) = 0.00;
    }
    
    descr->type_num = self->descriptor->type_num;
    descr->type = self->descriptor->type;
    descr->elsize = self->descriptor->elsize;

    if(axis == NULL) {
        descr->numElements = 1;
        ret = CArray_NewFromDescr_int(ret, descr, 0, NULL, NULL, NULL, 0, NULL, 1, 0);
        CArray_Data_alloc(ret);
        if(rtype == TYPE_INTEGER_INT) {
            for(i = 0; i < CArray_DESCR(self)->numElements; i++) {
                *((int*)total) += ((int*)CArray_DATA(self))[i];
            }
            ((int*)CArray_DATA(ret))[0] = *((int *)total);
        }
        if(rtype == TYPE_DOUBLE_INT) {
            for(i = 0; i < CArray_DESCR(self)->numElements; i++) {
                *((double*)total) += ((double*)CArray_DATA(self))[i];
            }
            ((double*)CArray_DATA(ret))[0] = *((double *)total);
        }
    }
    if(axis != NULL) {
        int * new_dimensions = (int*)emalloc((self->ndim - 1) * sizeof(int));    
        for(i = 0; i < self->ndim; i++) {
            if(i != *axis) {
                new_dimensions[j] = self->dimensions[i];
                j++;
            }         
        }      
        int num_elements = new_dimensions[0];
        int * strides = CArray_Generate_Strides(new_dimensions, self->ndim-1, self->descriptor->type);
        for(i = 1; i < self->ndim-1; i++) {
            num_elements *= new_dimensions[i];
        }
        descr->numElements = num_elements;

        CArray_Data_alloc(ret);
        
        if(rtype == TYPE_INTEGER_INT) {
            for(i = 0; i < num_elements; i++) {
                ((int*)CArray_DATA(ret))[i] = 0;
            }
            ret = CArray_NewFromDescr_int(ret, descr, self->ndim-1, new_dimensions, strides, CArray_DATA(ret), 0, NULL, 1, 0);   
        }
    }
    add_to_buffer(out_ptr, *ret, sizeof(*ret));
    efree(total);
    return ret;
}