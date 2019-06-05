#include "ctors.h"
#include "carray.h"
#include "alloc.h"
#include "convert.h"
#include "convert_type.h"

int
setArrayFromSequence(CArray *a, CArray *s, int dim, int offset)
{
    int i, slen;
    int res = -1;

    /* INCREF on entry DECREF on exit */
    CArray_INCREF(s);

    if (dim > a->ndim) {
        throw_valueerror_exception("setArrayFromSequence: sequence/array dimensions mismatch.");
        goto fail;
    }

    slen = CArray_SIZE(s);
    if (slen != a->dimensions[dim]) {
        throw_valueerror_exception("setArrayFromSequence: sequence/array shape mismatch.");
        goto fail;
    }

    for (i = 0; i < slen; i++) {
        CArray *o = CArray_Slice_Index(s, i, NULL);
        if ((a->ndim - dim) > 1) {
            res = setArrayFromSequence(a, o, dim+1, offset);
        }
        else {
            res = a->descriptor->f->setitem(o->data, (a->data + offset), a);
        }
        CArray_DECREF(o);
        CArray_DECREF(s);
        CArrayDescriptor_FREE(o->descriptor);
        CArray_Free(o);
        if (res < 0) {
            goto fail;
        }
        offset += a->strides[dim];
    }
    CArray_DECREF(s);
    return 0;

fail:
    CArray_DECREF(s);
    return res;
}

int
CArray_AssignFromSequence(CArray *self, CArray *v)
{
    if (CArray_NDIM(self) == 0) {
        throw_valueerror_exception("assignment to 0-d array");
        return -1;
    }
    return setArrayFromSequence(self, v, 0, 0);
}

CArray *
CArray_FromArray(CArray *arr, CArrayDescriptor *newtype, int flags)
{

    CArray *ret = NULL;
    int itemsize;
    int copy = 0;
    int arrflags;
    CArrayDescriptor *oldtype;
    CARRAY_CASTING casting = CARRAY_SAFE_CASTING;

    oldtype = CArray_DESCR(arr);
    if (newtype == NULL) {
        /*
         * Check if object is of array with Null newtype.
         * If so return it directly instead of checking for casting.
         */
        if (flags == 0) {
            CArray_INCREF(arr);
            return (CArray *)arr;
        }
        newtype = oldtype;
        CArrayDescriptor_INCREF(oldtype);
    }
    itemsize = newtype->elsize;
    if (itemsize == 0) {
        CArray_DESCR_REPLACE(newtype);
        if (newtype == NULL) {
            return NULL;
        }
        newtype->elsize = oldtype->elsize;
        itemsize = newtype->elsize;
    }

    /* If the casting if forced, use the 'unsafe' casting rule */
    if (flags & CARRAY_ARRAY_FORCECAST) {
        casting = CARRAY_UNSAFE_CASTING;
    }

    /* Raise an error if the casting rule isn't followed */
    if (!CArray_CanCastArrayTo(arr, newtype, casting)) {
        throw_valueerror_exception("Cannot cast array data according to the rule");
        CArrayDescriptor_DECREF(newtype);
        return NULL;
    }

    arrflags = CArray_FLAGS(arr);
    /* If a guaranteed copy was requested */
    copy = (flags & CARRAY_ARRAY_ENSURECOPY) ||
           /* If C contiguous was requested, and arr is not */
           ((flags & CARRAY_ARRAY_C_CONTIGUOUS) &&
            (!(arrflags & CARRAY_ARRAY_C_CONTIGUOUS))) ||
           /* If an aligned array was requested, and arr is not */
           ((flags & CARRAY_ARRAY_ALIGNED) &&
            (!(arrflags & CARRAY_ARRAY_ALIGNED))) ||
           /* If a Fortran contiguous array was requested, and arr is not */
           ((flags & CARRAY_ARRAY_F_CONTIGUOUS) &&
            (!(arrflags & CARRAY_ARRAY_F_CONTIGUOUS))) ||
           /* If a writeable array was requested, and arr is not */
           ((flags & CARRAY_ARRAY_WRITEABLE) &&
            (!(arrflags & CARRAY_ARRAY_WRITEABLE))) ||
           !CArray_EquivTypes(oldtype, newtype);

    if (copy) {
        CARRAY_ORDER order = CARRAY_KEEPORDER;
        int subok = 1;

        /* Set the order for the copy being made based on the flags */
        if (flags & CARRAY_ARRAY_F_CONTIGUOUS) {
            order = CARRAY_FORTRANORDER;
        }
        else if (flags & CARRAY_ARRAY_C_CONTIGUOUS) {
            order = CARRAY_CORDER;
        }

        if ((flags & CARRAY_ARRAY_ENSUREARRAY)) {
            subok = 0;
        }

        ret = CArray_NewLikeArray(arr, order, newtype, subok);

        if (ret == NULL) {
            return NULL;
        }

        if (CArray_CopyInto(ret, arr) < 0) {
            CArray_DECREF(ret);
            return NULL;
        }

        if (flags & CARRAY_ARRAY_UPDATEIFCOPY)  {
            ret->flags |= CARRAY_ARRAY_UPDATEIFCOPY;
            ret->base = arr;
            ret->flags &= ~CARRAY_ARRAY_WRITEABLE;
            CArray_INCREF(arr);
        }

        //CArray_Print(ret);
    }
        /*
         * If no copy then take an appropriate view if necessary, or
         * just return a reference to ret itself.
         */
    else {
        CArray_INCREF(arr);
        ret = arr;
    }

    return ret;
}

