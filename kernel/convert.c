//
// Created by Henrique Borba on 25/11/2018.
//

#include "convert.h"
#include "carray.h"
#include "alloc.h"

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