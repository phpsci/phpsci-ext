//
// Created by Henrique Borba on 25/11/2018.
//

#include "convert.h"
#include "carray.h"
#include "alloc.h"
#include "buffer.h"

/**
 * Slice CArray
 * 
 * @todo Handle Exceptions (invalid index, etc)
 **/
CArray *
CArray_Slice_Index(CArray * self, int index, MemoryPointer * out)
{
    CArray * ret = NULL;
    CArrayDescriptor * subarray_descr;
    int * new_dimensions, * new_strides;
    int new_num_elements = 0;
    int nd, i, flags;
    ret = (CArray *)emalloc(sizeof(CArray));

    subarray_descr = (CArrayDescriptor *)emalloc(sizeof(CArrayDescriptor));
    nd = CArray_NDIM(self) - 1;
    new_dimensions = (int*)emalloc(nd * sizeof(int));
    
    for(i = 1; i < CArray_NDIM(self); i++) {
        new_dimensions[i-1] = self->dimensions[i];
    }
    subarray_descr->elsize = CArray_DESCR(self)->elsize;
    subarray_descr->type = CArray_DESCR(self)->type;
    subarray_descr->type_num = CArray_DESCR(self)->type_num;

    new_strides = CArray_Generate_Strides(new_dimensions, nd, self->descriptor->type);
    
    new_num_elements = self->dimensions[nd];
    
    for(i = nd-1; i > 0; i--) {
        new_num_elements = new_num_elements * CArray_DIMS(self)[i];
    }
    subarray_descr->numElements = new_num_elements;
    ret->descriptor = subarray_descr;
    flags = CArray_FLAGS(self);
    ret = (CArray *)CArray_NewFromDescr_int(
            ret, subarray_descr,
            nd, new_dimensions, new_strides,
            (CArray_DATA(self) + (index * self->strides[0])),
            flags, self,
            0, 1);

    add_to_buffer(out, *ret, sizeof(*ret));        
    return ret;        
}

/**
 * @param self
 * @param target
 * @return
 */
CArray *
CArray_View(CArray *self)
{
    CArray *ret = NULL;
    CArrayDescriptor *dtype;
    CArray *subtype;
    int flags;

    dtype = CArray_DESCR(self);

    flags = CArray_FLAGS(self);

    CArray_INCREF(self);
    ret = (CArray *)CArray_NewFromDescr_int(
            self, dtype,
            CArray_NDIM(self), CArray_DIMS(self), CArray_STRIDES(self),
            CArray_DATA(self),
            flags, self,
            0, 1);

    return ret;
}

/**
 * @param obj
 * @param order
 * @return
 */
CArray *
CArray_NewCopy(CArray *obj, CARRAY_ORDER order)
{
    CArray * ret;

    ret = (CArray *)CArray_NewLikeArray(obj, order, NULL, 1);
    if (ret == NULL) {
        return NULL;
    }

    return ret;
}