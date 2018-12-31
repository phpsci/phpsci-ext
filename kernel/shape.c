//
// Created by Henrique Borba on 25/11/2018.
//

#include "shape.h"
#include "carray.h"
#include "common/exceptions.h"
#include "convert.h"
#include "alloc.h"
#include "flagsobject.h"
#include "buffer.h"

/*
 * attempt to reshape an array without copying data
 *
 * The requested newdims are not checked, but must be compatible with
 * the size of self, which must be non-zero. Other than that this
 * function should correctly handle all reshapes, including axes of
 * length 1. Zero strides should work but are untested.
 *
 * If a copy is needed, returns 0
 * If no copy is needed, returns 1 and fills newstrides
 *     with appropriate strides
 *
 * The "is_f_order" argument describes how the array should be viewed
 * during the reshape, not how it is stored in memory (that
 * information is in CArray_STRIDES(self)).
 *
 * If some output dimensions have length 1, the strides assigned to
 * them are arbitrary. In the current implementation, they are the
 * stride of the next-fastest index.
 */
static int
_attempt_nocopy_reshape(CArray *self, int newnd, int* newdims,
                        int *newstrides, int is_f_order)
{
    int oldnd;
    int olddims[CARRAY_MAXDIMS];
    int oldstrides[CARRAY_MAXDIMS];
    int last_stride;
    int oi, oj, ok, ni, nj, nk;

    oldnd = 0;
    /*
     * Remove axes with dimension 1 from the old array. They have no effect
     * but would need special cases since their strides do not matter.
     */
    for (oi = 0; oi < CArray_NDIM(self); oi++) {
        if (CArray_DIMS(self)[oi]!= 1) {
            olddims[oldnd] = CArray_DIMS(self)[oi];
            oldstrides[oldnd] = CArray_STRIDES(self)[oi];
            oldnd++;
        }
    }

    /* oi to oj and ni to nj give the axis ranges currently worked with */
    oi = 0;
    oj = 1;
    ni = 0;
    nj = 1;
    while (ni < newnd && oi < oldnd) {
        int np = newdims[ni];
        int op = olddims[oi];

        while (np != op) {
            if (np < op) {
                /* Misses trailing 1s, these are handled later */
                np *= newdims[nj++];
            } else {
                op *= olddims[oj++];
            }
        }

        /* Check whether the original axes can be combined */
        for (ok = oi; ok < oj - 1; ok++) {
            if (is_f_order) {
                if (oldstrides[ok+1] != olddims[ok]*oldstrides[ok]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
            else {
                /* C order */
                if (oldstrides[ok] != olddims[ok+1]*oldstrides[ok+1]) {
                    /* not contiguous enough */
                    return 0;
                }
            }
        }

        /* Calculate new strides for all axes currently worked with */
        if (is_f_order) {
            newstrides[ni] = oldstrides[oi];
            for (nk = ni + 1; nk < nj; nk++) {
                newstrides[nk] = newstrides[nk - 1]*newdims[nk - 1];
            }
        }
        else {
            /* C order */
            newstrides[nj - 1] = oldstrides[oj - 1];
            for (nk = nj - 1; nk > ni; nk--) {
                newstrides[nk - 1] = newstrides[nk]*newdims[nk];
            }
        }
        ni = nj++;
        oi = oj++;
    }

    /*
     * Set strides corresponding to trailing 1s of the new shape.
     */
    if (ni >= 1) {
        last_stride = newstrides[ni - 1];
    }
    else {
        last_stride = CArray_ITEMSIZE(self);
    }
    if (is_f_order) {
        last_stride *= newdims[ni - 1];
    }
    for (nk = ni; nk < newnd; nk++) {
        newstrides[nk] = last_stride;
    }

    return 1;
}

/**
 * CArray Transpose
 **/ 
CArray *
CArray_Transpose(CArray * target, CArray_Dims * permute, MemoryPointer * ptr)
{
    int * axes;
    int i, n;
    int * permutation = NULL, * reverse_permutation = NULL;
    CArray * ret = NULL;
    int flags;

    ret = (CArray *)emalloc(sizeof(CArray));

    if(permute == NULL) {
        n = CArray_NDIM(target);
        permutation = (int *)emalloc(n * sizeof(int));
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    if(permute != NULL) {
        n = permute->len;
        axes = permute->ptr;
        permutation = (int *)emalloc(n * sizeof(int));
        reverse_permutation = (int *)emalloc(n * sizeof(int));
        if(n != CArray_NDIM(target)) {
            throw_axis_exception("axes don't match array");
            return NULL;
        }

        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }

        for (i = 0; i < n; i++) {
            int axis = axes[i];
            if (check_and_adjust_axis(&axis, CArray_NDIM(target)) < 0) {
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                throw_axis_exception("repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
    }

    flags = CArray_FLAGS(target);
    CArrayDescriptor_INCREF(CArray_DESCR(target));

    ret = CArray_NewFromDescrAndBase(
            ret, CArray_DESCR(target),
            n, CArray_DIMS(target), NULL, CArray_DATA(target),
            flags, target);

    if (ret == NULL) {
        return NULL;
    }
    
    for (i = 0; i < n; i++) {
        CArray_DIMS(ret)[i] = CArray_DIMS(target)[permutation[i]];
        CArray_STRIDES(ret)[i] = CArray_STRIDES(target)[permutation[i]];
    }
    ret->flags |= CARRAY_ARRAY_F_CONTIGUOUS;
    CArray_UpdateFlags(ret, CARRAY_ARRAY_C_CONTIGUOUS | CARRAY_ARRAY_F_CONTIGUOUS | CARRAY_ARRAY_ALIGNED);

    efree(permutation);
    if(reverse_permutation != NULL) {
        efree(reverse_permutation);
    }

    add_to_buffer(ptr, ret, sizeof(CArray));
    return ret;
}

/**
 * @param self
 * @param newdims
 * @param order
 * @return
 */
CArray *
CArray_Newshape(CArray * self, int *newdims, int new_ndim, CARRAY_ORDER order, MemoryPointer * ptr)
{
    int i;
    int *dimensions = newdims;
    CArray *ret = NULL;
    int ndim = new_ndim;
    int same;
    int *strides = NULL;
    int newstrides[CARRAY_MAXDIMS];
    int flags;

    if (order == CARRAY_ANYORDER) {
        order = CArray_ISFORTRAN(self);
    }
    else if (order == CARRAY_KEEPORDER) {
        php_printf("order 'K' is not permitted for reshaping");
        return NULL;
    }

    if (ndim == CArray_NDIM(self)) {
        same = 1;
        i = 0;
        while (same && i < ndim) {
            if (CArray_DIM(self,i) != dimensions[i]) {
                same=0;
            }
            i++;
        }
        if (same) {
            ret = CArray_View(self);
            add_to_buffer(ptr, ret, sizeof(CArray));
            return ret;
        }
    }

    CArray_INCREF(self);

    if ((CArray_SIZE(self) > 1) &&
        ((order == CARRAY_CORDER && !CArray_IS_C_CONTIGUOUS(self)) ||
         (order == CARRAY_FORTRANORDER && !CArray_IS_F_CONTIGUOUS(self)))) {

        int success = 0;
        success = _attempt_nocopy_reshape(self, ndim, dimensions, newstrides, order);
        if (success) {
            /* no need to copy the array after all */
            strides = newstrides;
        }
        else {
            CArray * newcopy;
            newcopy = CArray_NewCopy(self, order);
            CArray_DECREF(self);
            if (newcopy == NULL) {
                return NULL;
            }
            self = newcopy;
        }
    }

    /* Make sure the flags argument is set. */
    flags = CArray_FLAGS(self);
    if (ndim > 1) {
        if (order == CARRAY_FORTRANORDER) {
            flags &= ~CARRAY_ARRAY_C_CONTIGUOUS;
            flags |= CARRAY_ARRAY_F_CONTIGUOUS;
        }
        else {
            flags &= ~CARRAY_ARRAY_F_CONTIGUOUS;
            flags |= CARRAY_ARRAY_C_CONTIGUOUS;
        }
    }

    if(ret == NULL) {
        ret = (CArray *)emalloc(sizeof(CArray));
    }

    CArrayDescriptor_INCREF(CArray_DESCR(self));

    strides = CArray_Generate_Strides(newdims, ndim, self->descriptor->type);
    
    ret =   CArray_NewFromDescr_int(
            ret, CArray_DESCR(self),
            ndim, newdims, strides, CArray_DATA(self),
            flags, self, 0, 1);
              
    CArray_DECREF(self);
    add_to_buffer(ptr, ret, sizeof(CArray));

    return ret;
}

/**
 * @return
 */
CArray *
CArray_SwapAxes(CArray * ap, int a1, int a2, MemoryPointer * out)
{
    CArray_Dims new_axes;
    int * dims;
    int n, i, val;
    CArray * ret;

    if (a1 == a2) {
        CArray_INCREF(ap);
        return ap;
    }

    n = CArray_NDIM(ap);
    if (n <= 1) {
        CArray_INCREF(ap);
        return ap;
    }

    if (a1 < 0) {
        a1 += n;
    }
    if (a2 < 0) {
        a2 += n;
    }
    if ((a1 < 0) || (a1 >= n)) {
        throw_valueerror_exception("bad axis1 argument to swapaxes");
        return NULL;
    }

    if ((a2 < 0) || (a2 >= n)) {
        throw_valueerror_exception("bad axis2 argument to swapaxes");
        return NULL;
    }

    dims = emalloc(n * sizeof(int));
    new_axes.ptr = dims;
    new_axes.len = n;

    for (i = 0; i < n; i++) {
        if (i == a1) {
            val = a2;
        }
        else if (i == a2) {
            val = a1;
        }
        else {
            val = i;
        }
        new_axes.ptr[i] = val;
    }
    ret = CArray_Transpose(ap, &new_axes, out);
    efree(dims);
    return ret;
}