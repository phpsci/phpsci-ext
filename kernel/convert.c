#include "convert.h"
#include "carray.h"
#include "alloc.h"
#include "buffer.h"
#include "common/exceptions.h"
#include "scalar.h"
#include "assign_scalar.h"

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
    ret = (CArray *)ecalloc(1, sizeof(CArray));
   
    subarray_descr = (CArrayDescriptor *)ecalloc(1, sizeof(CArrayDescriptor));
    nd = CArray_NDIM(self) - 1;
    new_dimensions = (int*)emalloc(nd * sizeof(int));
    
    for(i = 1; i < CArray_NDIM(self); i++) {
        new_dimensions[i-1] = self->dimensions[i];
    }
    subarray_descr->elsize = CArray_DESCR(self)->elsize;
    subarray_descr->type = CArray_DESCR(self)->type;
    subarray_descr->type_num = CArray_DESCR(self)->type_num;
    subarray_descr->alignment = 0;

    new_strides  = (int*)emalloc(nd * sizeof(int));
    for(i = 1; i < CArray_NDIM(self); i++) {
        new_strides[i-1] = self->strides[i];
    }
    
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
            
    add_to_buffer(out, ret, sizeof(*ret));
    efree(new_dimensions);
    efree(new_strides);
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
    CArray *ret = emalloc(sizeof(CArray));
    CArrayDescriptor *dtype;
    CArray *subtype;
    int flags;

    dtype = CArray_DESCR(self);

    flags = CArray_FLAGS(self);

    ret = (CArray *)CArray_NewFromDescr_int(
            ret, dtype,
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
    ret = (CArray *)emalloc(sizeof(CArray));
    ret = (CArray *)CArray_NewLikeArray(obj, order, NULL, 1);

    return ret;
}

int
CArray_CanCastSafely(int fromtype, int totype)
{
    CArrayDescriptor *from, *to;
    int felsize, telsize;

    if (fromtype == totype) {
        return 1;
    }
    if (fromtype == TYPE_BOOL_INT) {
        return 1;
    }
    if (totype == TYPE_BOOL_INT) {
        return 0;
    }

    from = CArray_DescrFromType(fromtype);
    /*
     * cancastto is a NPY_NOTYPE terminated C-int-array of types that
     * the data-type can be cast to safely.
     */
    /**if (from->f->cancastto) {
        int *curtype;
        curtype = from->f->cancastto;
        while (*curtype != NPY_NOTYPE) {
            if (*curtype++ == totype) {
                return 1;
            }
        }
    }**/
    
    to = CArray_DescrFromType(totype);
    telsize = to->elsize;
    felsize = from->elsize;
    CArrayDescriptor_DECREF(from);
    CArrayDescriptor_DECREF(to);

    switch(fromtype) {
        default:
            return 0;
    }
}

int
CArray_CanCastTo(CArrayDescriptor *from, CArrayDescriptor *to)
{
    int fromtype = from->type_num;
    int totype = to->type_num;
    int ret;

    ret = CArray_CanCastSafely(fromtype, totype);

    if (ret) {
        /* Check String and Unicode more closely */
        if (fromtype == TYPE_STRING_INT) {
            if (totype == TYPE_STRING_INT) {
                ret = (from->elsize <= to->elsize);
            }
        }
        /*
         * TODO: If totype is STRING or unicode
         * see if the length is long enough to hold the
         * stringified value of the object.
         */
    }
    return ret;
}

CArray_VectorUnaryFunc *
CArray_GetCastFunc(CArrayDescriptor *descr, int type_num)
{
    CArray_VectorUnaryFunc *castfunc = NULL;

    castfunc = descr->f->cast[type_num];
    
    if (NULL == castfunc) {
        throw_valueerror_exception("No cast function available.");
        return NULL;
    }
    return castfunc;
}

int
CArray_CastTo(CArray *out, CArray *mp)
{
    int simple;
    int same;
    CArray_VectorUnaryFunc *castfunc = NULL;
    int mpsize = CArray_SIZE(mp);
    int iswap, oswap;

    if (mpsize == 0) {
        return 0;
    }
    if (!CArray_ISWRITEABLE(out)) {
        throw_valueerror_exception("output array is not writeable");
        return -1;
    }

    castfunc = CArray_GetCastFunc(CArray_DESCR(mp), CArray_DESCR(out)->type_num);
    if (castfunc == NULL) {
        return -1;
    }

    same = CArray_SAMESHAPE(out, mp);
    simple = same && ((CArray_ISCARRAY_RO(mp) && CArray_ISCARRAY(out)) ||
                      (CArray_ISFARRAY_RO(mp) && CArray_ISFARRAY(out)));

    if (simple) {
        castfunc(mp->data, out->data, mpsize, mp, out);
        return 0;
    }         
}

int
CArray_FillWithScalar(CArray * arr, CArrayScalar * sc)
{
    CArrayDescriptor * dtype = NULL;
    long long value_buffer[4];
    void * value = NULL;
    int retcode = 0;
    
    dtype = CArray_DescrFromScalar(sc);
    value = scalar_value(sc, dtype);
    if (value == NULL) {
        CArrayDescriptor_FREE(dtype);
        return -1;
    }
    
    /* Use the value pointer we got if possible */
    if (value != NULL) {
        /* TODO: switch to SAME_KIND casting */
        retcode = CArray_AssignRawScalar(arr, dtype, value, NULL, CARRAY_UNSAFE_CASTING);
        
        CArrayDescriptor_FREE(dtype);
        return retcode;
    }

    CArrayDescriptor_FREE(dtype);
}