#include "shape.h"
#include "carray.h"
#include "common/exceptions.h"
#include "convert.h"
#include "alloc.h"
#include "flagsobject.h"
#include "buffer.h"
#include "conversion_utils.h"

/*
 * Sorts items so stride is descending, because C-order
 * is the default in the face of ambiguity.
 */
static int _carray_stride_sort_item_comparator(const void *a, const void *b)
{
    int astride = ((const ca_stride_sort_item *)a)->stride,
            bstride = ((const ca_stride_sort_item *)b)->stride;

    /* Sort the absolute value of the strides */
    if (astride < 0) {
        astride = -astride;
    }
    if (bstride < 0) {
        bstride = -bstride;
    }

    if (astride == bstride) {
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
        int aperm = ((const ca_stride_sort_item *)a)->perm,
                bperm = ((const ca_stride_sort_item *)b)->perm;
        return (aperm < bperm) ? -1 : 1;
    }
    if (astride > bstride) {
        return -1;
    }
    return 1;
}

/*
 * This function populates the first ndim elements
 * of strideperm with sorted descending by their absolute values.
 * For example, the stride array (4, -2, 12) becomes
 * [(2, 12), (0, 4), (1, -2)].
 */
void
CArray_CreateSortedStridePerm(int ndim, int *strides,
                              ca_stride_sort_item *out_strideperm)
{
    int i;

    /* Set up the strideperm values */
    for (i = 0; i < ndim; ++i) {
        out_strideperm[i].perm = i;
        out_strideperm[i].stride = strides[i];
    }

    /* Sort them */
    qsort(out_strideperm, ndim, sizeof(ca_stride_sort_item),
                                    &_carray_stride_sort_item_comparator);
}

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

/*
 *
 * Removes the axes flagged as True from the array,
 * modifying it in place. If an axis flagged for removal
 * has a shape entry bigger than one, this effectively selects
 * index zero for that axis.
 *
 * WARNING: If an axis flagged for removal has a shape equal to zero,
 *          the array will point to invalid memory. The caller must
 *          validate this!
 *          If an axis flagged for removal has a shape larger than one,
 *          the aligned flag (and in the future the contiguous flags),
 *          may need explicit update.
 *
 * For example, this can be used to remove the reduction axes
 * from a reduction result once its computation is complete.
 */
void
CArray_RemoveAxesInPlace(CArray *arr, int *flags)
{
    int *shape = CArray_DIMS(arr), *strides = CArray_STRIDES(arr);
    int idim, ndim = CArray_NDIM(arr), idim_out = 0;

    /* Compress the dimensions and strides */
    for (idim = 0; idim < ndim; ++idim) {
        if (!flags[idim]) {
            shape[idim_out] = shape[idim];
            strides[idim_out] = strides[idim];
            ++idim_out;
        }
    }

    /* The final number of dimensions */

    arr->ndim = idim_out;

    CArray_UpdateFlags(arr, CARRAY_ARRAY_C_CONTIGUOUS | CARRAY_ARRAY_F_CONTIGUOUS);
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
        throw_valueerror_exception("order 'K' is not permitted for reshaping");
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
            if(ptr != NULL) { 
                add_to_buffer(ptr, ret, sizeof(CArray));
            }
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
    
    ret =   CArray_NewFromDescr_int(
            ret, CArray_DESCR(self),
            ndim, newdims, strides, CArray_DATA(self),
            flags, self, 0, 1);
     
            
    CArrayDescriptor_INCREF(CArray_DESCR(self));
    if(ptr != NULL) {
        add_to_buffer(ptr, ret, sizeof(CArray));
    }
    CArray_DECREF(self);
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

/**
 * Roll the specified axis backwards, until it lies in a given position.
 *
 * @param arr
 * @param axis
 * @param start
 * @param out
 * @return
 */
CArray *
CArray_Rollaxis(CArray * arr, int axis, int start, MemoryPointer * out)
{
    int i, tmp_val, j = 0;
    int n = CArray_NDIM(arr);
    CArray * rtn = NULL;
    CArray_Dims permute;

    if (check_and_adjust_axis_msg(&axis, n) < 0) {
        return NULL;
    }

    if (start < 0) {
        start += n;
    }

    if (!(0 <= start < n + 1)) {
        throw_axis_exception("Invalid start option for Rollaxis");
    }

    if (axis < start) {
        start -= 1;
    }

    if (axis == start) {
        CArrayDescriptor * new_descr = CArray_DescrNew(CArray_DESCR(arr));
        CArrayDescriptor_INCREF(CArray_DESCR(arr));
        rtn = CArray_View(arr);
    }

    if (axis != start) {
        int * axes = emalloc(sizeof(int) * n);
        for (i = 0; i < n; i++) {
            if (i != axis) {
                axes[j] = i;
                j++;
            }
        }

        for (i = n-1; i > start; i--) {
            axes[i] = axes[i - 1];
        }

        axes[start] = axis;

        permute.ptr = axes;
        permute.len = n;
        return CArray_Transpose(arr, &permute, out);
    }

    if (out != NULL) {
        add_to_buffer(out, rtn, sizeof(CArray));
    }

    return rtn;
}

/*
 * Ravel
 * Returns a contiguous array
 */
CArray *
CArray_Ravel(CArray *arr, CARRAY_ORDER order)
{
    CArray_Dims newdim = {NULL,1};
    int val[1] = {CArray_SIZE(arr)};

    newdim.ptr = val;

    if (order == CARRAY_KEEPORDER) {
        /* This handles some corner cases, such as 0-d arrays as well */
        if (CArray_IS_C_CONTIGUOUS(arr)) {
            order = CARRAY_CORDER;
        }
        else if (CArray_IS_F_CONTIGUOUS(arr)) {
            order = CARRAY_FORTRANORDER;
        }
    }
    else if (order == CARRAY_ANYORDER) {
        order = CArray_ISFORTRAN(arr) ? CARRAY_FORTRANORDER : CARRAY_CORDER;
    }
    
    if (order == CARRAY_CORDER && CArray_IS_C_CONTIGUOUS(arr)) {
        return CArray_Newshape(arr, newdim.ptr, 1, CARRAY_CORDER, NULL);
    }
}

/**
 * @todo FIX INCREF and DECREF weird behaviour
 **/ 
CArray *
CArray_atleast1d(CArray * self, MemoryPointer * out)
{
    int * dims;
    CArray * rtn;
    if (CArray_NDIM(self) >= 1) {
        rtn = CArray_View(self);
        if (out != NULL) {
            add_to_buffer(out, rtn, sizeof(CArray));
        }
        return rtn;
    }
    dims = emalloc(sizeof(int));
    dims[0] = 1;
    rtn = CArray_Newshape(self, dims, 1, CARRAY_CORDER, out);
    efree(dims);
}

/**
 * @todo FIX INCREF and DECREF weird behaviour
 **/ 
CArray *
CArray_atleast2d(CArray * self, MemoryPointer * out)
{
    int * dims;
    CArray * rtn;
    if (CArray_NDIM(self) >= 2) {
        rtn = CArray_View(self);
        CArrayDescriptor_INCREF(CArray_DESCR(rtn));
        if (out != NULL) {
            add_to_buffer(out, rtn, sizeof(CArray));
        }
        CArrayDescriptor_INCREF(CArray_DESCR(rtn));
        return rtn;
    }
    dims = emalloc(sizeof(int) * 2);
    dims[0] = 1;
    if(CArray_NDIM(self) == 0) {
        dims[1] = 1;
    } else {
        dims[1] = CArray_DIMS(self)[0];
    }
    
    rtn = CArray_Newshape(self, dims, 2, CARRAY_CORDER, out);
    efree(dims);
}

CArray *
CArray_atleast3d(CArray * self, MemoryPointer * out)
{
    int * dims;
    CArray * rtn;
    if (CArray_NDIM(self) >= 3) {
        rtn = CArray_View(self);
        CArrayDescriptor_INCREF(CArray_DESCR(rtn));
        if (out != NULL) {
            add_to_buffer(out, rtn, sizeof(CArray));
        }
        CArrayDescriptor_INCREF(CArray_DESCR(rtn));
        return rtn;
    }
    dims = emalloc(sizeof(int) * 3);
    dims[0] = 1;
    if(CArray_NDIM(self) == 0) {
        dims[1] = 1;
        dims[2] = 1;
    } 
    if(CArray_NDIM(self) == 1) {
        dims[1] = 1;
        dims[2] = CArray_DIMS(self)[0];
    } 
    if(CArray_NDIM(self) == 2) {
        dims[1] = CArray_DIMS(self)[0];
        dims[2] = CArray_DIMS(self)[1];
    } 
    
    rtn = CArray_Newshape(self, dims, 3, CARRAY_CORDER, out);
    efree(dims);
}

CArray *
CArray_SqueezeSelected(CArray * self, int *axis_flags)
{
    CArray *ret;
    int idim, ndim, any_ones;
    int *shape;

    ndim = CArray_NDIM(self);
    shape = CArray_DIMS(self);
 
    /* Verify that the axes requested are all of size one */
    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim] != 0) {
            if (shape[idim] == 1) {
                any_ones = 1;
            }
            else {
                throw_valueerror_exception("cannot select an axis to squeeze out which has size not equal to one");
                return NULL;
            }
        }
    }

    /* If there were no axes to squeeze out, return the same array */
    if (!any_ones) {
        CArray_INCREF(self);
        return self;
    }

    
    ret = CArray_View(self);
    if (ret == NULL) {
        return NULL;
    }
    CArrayDescriptor_INCREF(CArray_DESCR(ret));
    CArray_RemoveAxesInPlace(ret, axis_flags);

    return ret;
}

CArray *
CArray_Squeeze(CArray * self, int axis, MemoryPointer * out)
{
    CArray *ret;
    int *axis_in = NULL;
    int * axis_flags =  ecalloc(sizeof(int), CArray_NDIM(self));
    if (axis == INT_MAX) {
        int * unit_dims = emalloc(sizeof(int) * CArray_NDIM(self));
        int idim, ndim, any_ones;
        int *shape;

        ndim = CArray_NDIM(self);
        shape = CArray_DIMS(self);

        any_ones = 0;
        for (idim = 0; idim < ndim; ++idim) {
            if (shape[idim] == 1) {
                unit_dims[idim] = 1;
                any_ones = 1;
            }
            else {
                unit_dims[idim] = 0;
            }
        }

        /* If there were no ones to squeeze out, return the same array */
        if (!any_ones) {
            CArray_INCREF(self);
            return self;
        }

        ret = CArray_View(self);
        if (ret == NULL) {
            return NULL;
        }
        
        CArray_RemoveAxesInPlace(ret, unit_dims);
        efree(unit_dims);
        efree(axis_flags);
    } else {
        axis_in = emalloc(sizeof(int));
        *axis_in = axis;
        if (CArray_ConvertMultiAxis(axis_in, CArray_NDIM(self), axis_flags) != CARRAY_SUCCEED) {
            return NULL;
        }

        ret = CArray_SqueezeSelected(self, axis_flags);
        
    }
    
    if(out != NULL && ret != NULL) {
        add_to_buffer(out, ret, sizeof(CArray));
    } else {
        return NULL;
    }
}