#include "carray.h"
#include "buffer.h"
#include "common/exceptions.h"
#include "common/common.h"

/**
 * FAST TAKE
 */ 
int
INT_fasttake(int *dest, int *src, int *indarray,
                    int nindarray, int n_outer,
                    int m_middle, int nelem,
                    CARRAY_CLIPMODE clipmode)
{
    int i, j, k, tmp;
    switch(clipmode) {
    case CARRAY_RAISE:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                /*
                 * We don't know what axis we're operating on,
                 * so don't report it in case of an error.
                 */
                if (check_and_adjust_index(&tmp, nindarray, -1) < 0) {
                    return 1;
                }
                if (CARRAY_UNLIKELY(nelem == 1)) {
                    *dest++ = *(src + tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src + tmp*nelem + k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    case CARRAY_WRAP:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += nindarray;
                    }
                }
                else if (tmp >= nindarray) {
                    while (tmp >= nindarray) {
                        tmp -= nindarray;
                    }
                }
                if (CARRAY_UNLIKELY(nelem == 1)) {
                    *dest++ = *(src+tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src+tmp*nelem+k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    case CARRAY_CLIP:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= nindarray) {
                    tmp = nindarray - 1;
                }
                if (CARRAY_UNLIKELY(nelem == 1)) {
                    *dest++ = *(src + tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src + tmp*nelem + k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    }
    return 0;
}
int
DOUBLE_fasttake(double *dest, double *src, int *indarray,
                    int nindarray, int n_outer,
                    int m_middle, int nelem,
                    CARRAY_CLIPMODE clipmode)
{
    int i, j, k, tmp;

    switch(clipmode) {
    case CARRAY_RAISE:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                /*
                 * We don't know what axis we're operating on,
                 * so don't report it in case of an error.
                 */
                if (check_and_adjust_index(&tmp, nindarray, -1) < 0) {
                    return 1;
                }
                if (CARRAY_LIKELY(nelem == 1)) {
                    *dest++ = *(src + tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src + tmp*nelem + k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    case CARRAY_WRAP:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                if (tmp < 0) {
                    while (tmp < 0) {
                        tmp += nindarray;
                    }
                }
                else if (tmp >= nindarray) {
                    while (tmp >= nindarray) {
                        tmp -= nindarray;
                    }
                }
                if (CARRAY_LIKELY(nelem == 1)) {
                    *dest++ = *(src+tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src+tmp*nelem+k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    case CARRAY_CLIP:
        for (i = 0; i < n_outer; i++) {
            for (j = 0; j < m_middle; j++) {
                tmp = indarray[j];
                if (tmp < 0) {
                    tmp = 0;
                }
                else if (tmp >= nindarray) {
                    tmp = nindarray - 1;
                }
                if (CARRAY_LIKELY(nelem == 1)) {
                    *dest++ = *(src + tmp);
                }
                else {
                    for (k = 0; k < nelem; k++) {
                        *dest++ = *(src + tmp*nelem + k);
                    }
                }
            }
            src += nelem*nindarray;
        }
        break;
    }
    return 0;
}


/*
 * Diagonal
 */
CArray * 
CArray_Diagonal(CArray *self, int offset, int axis1, int axis2, MemoryPointer * rtn_ptr)
{
    int i, idim, ndim = CArray_NDIM(self);
    int *strides;
    int stride1, stride2, offset_stride;
    int *shape, dim1, dim2;

    char *data;
    int diag_size;
    CArrayDescriptor *dtype;
    CArray *ret = emalloc(sizeof(CArray));
    int ret_shape[CARRAY_MAXDIMS], ret_strides[CARRAY_MAXDIMS];

    if (ndim < 2) {
        throw_valueerror_exception("diag requires an array of at least two dimensions");
    }

    if (check_and_adjust_axis_msg(&axis1, ndim) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, ndim) < 0) {
        return NULL;
    }

    if (axis1 == axis2) {
        throw_valueerror_exception("axis1 and axis2 cannot be the same");
    }

    /* Get the shape and strides of the two axes */
    shape = CArray_DIMS(self);
    dim1 = shape[axis1];
    dim2 = shape[axis2];
    strides = CArray_STRIDES(self);
    stride1 = strides[axis1];
    stride2 = strides[axis2];

    /* Compute the data pointers and diag_size for the view */
    data = CArray_DATA(self);
    if (offset >= 0) {
        offset_stride = stride2;
        dim2 -= offset;
    }
    else {
        offset = -offset;
        offset_stride = stride1;
        dim1 -= offset;
    }
    diag_size = dim2 < dim1 ? dim2 : dim1;
    if (diag_size < 0) {
        diag_size = 0;
    }
    else {
        data += offset * offset_stride;
    }

    /* Build the new shape and strides for the main data */
    i = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (idim != axis1 && idim != axis2) {
            ret_shape[i] = shape[idim];
            ret_strides[i] = strides[idim];
            ++i;
        }
    }
    ret_shape[ndim-2] = diag_size;
    ret_strides[ndim-2] = stride1 + stride2;

    /* Create the diagonal view */
    dtype = CArray_DESCR(self);
    ret = CArray_NewFromDescrAndBase(
            ret, dtype, ndim-1, ret_shape, ret_strides, data,
            CArray_FLAGS(self), self);

    ret->flags &= ~CARRAY_ARRAY_WRITEABLE;
    ret->flags |= CARRAY_ARRAY_C_CONTIGUOUS | CARRAY_ARRAY_F_CONTIGUOUS;

    if (ret == NULL) {
        return NULL;
    }

    if(rtn_ptr != NULL) {
        
        add_to_buffer(rtn_ptr, ret, sizeof(CArray));
    }

    return ret;
}

/**
 * TakeFrom
 **/ 
CArray *
CArray_TakeFrom(CArray * target, CArray * indices0, int axis, 
                MemoryPointer * out_ptr, CARRAY_CLIPMODE clipmode)
{
    CArrayDescriptor * indices_type;
    CArray * out = NULL;
    CArrayDescriptor *dtype;
    CArray_FastTakeFunc *func;
    CArray *obj = NULL, *self, *indices;
    int nd, i, j, n, m, k, max_item, tmp, chunk, itemsize, nelem;
    int shape[CARRAY_MAXDIMS];
    char *src, *dest, *tmp_src;
    int err;
    int needs_refcounting;

    indices = NULL;
    
    self = (CArray *)CArray_CheckAxis(target, &axis, CARRAY_ARRAY_CARRAY_RO);

    
    if (self == NULL) {
        return NULL;
    }
    
    indices_type = CArray_DescrFromType(TYPE_INTEGER_INT);
    indices = CArray_FromCArray(indices0, indices_type, 0);

    if (indices == NULL) {
        goto fail;
    }
    

    n = m = chunk = 1;
    nd = CArray_NDIM(self) + CArray_NDIM(indices) - 1;
    
    for (i = 0; i < nd; i++) {
        if (i < axis) {
            shape[i] = CArray_DIMS(self)[i];
            n *= shape[i];
        }
        else {
            if (i < axis+CArray_NDIM(indices)) {
                shape[i] = CArray_DIMS(indices)[i-axis];
                m *= shape[i];
            }
            else {
                shape[i] = CArray_DIMS(self)[i-CArray_NDIM(indices)+1];
                chunk *= shape[i];
            }
        }
    }
    
    
    if (out == NULL) {
        if (obj == NULL) {
            obj = emalloc(sizeof(CArray));
        }
        dtype = CArray_DESCR(self);
        CArrayDescriptor_INCREF(dtype);
        obj = (CArray *)CArray_NewFromDescr(obj, dtype, nd, shape,
                                            NULL, NULL, 0, self);

        if (obj == NULL) {
            goto fail;
        }
    }
    else {
        int flags = CARRAY_ARRAY_CARRAY | CARRAY_ARRAY_WRITEBACKIFCOPY;

        if ((CArray_NDIM(out) != nd) ||
            !CArray_CompareLists(CArray_DIMS(out), shape, nd)) {
            throw_valueerror_exception("output array does not match result of ndarray.take");
            goto fail;
        }

        if (clipmode == CARRAY_RAISE) {
            /*
             * we need to make sure and get a copy
             * so the input array is not changed
             * before the error is called
             */
            flags |= CARRAY_ARRAY_ENSURECOPY;
        }
        dtype = CArray_DESCR(self);
        CArrayDescriptor_INCREF(dtype);
        obj = (CArray *)CArray_FromCArray(out, dtype, flags);
        if (obj == NULL) {
            goto fail;
        }
    }

    max_item = CArray_DIMS(self)[axis];
    nelem = chunk;
    itemsize = CArray_ITEMSIZE(obj);
    chunk = chunk * itemsize;
    src = CArray_DATA(self);
    dest = CArray_DATA(obj);
    needs_refcounting = CArrayDataType_REFCHK(CArray_DESCR(self));


    if ((max_item == 0) && (CArray_SIZE(obj) != 0)) {
        throw_indexerror_exception("cannot do a non-empty take from an empty axes.");
        goto fail;
    }

    func = CArray_DESCR(self)->f->fasttake;
    if (func == NULL) {
        goto fail;
    }
    else {
        /* no gil release, need it for error reporting */
        err = func(dest, src, (int *)(CArray_DATA(indices)),
                    max_item, n, m, nelem, clipmode);
        if (err) {
            goto fail;
        }
    }

    CArray_DECREF(indices);
    CArray_DECREF(self);
    if (out != NULL && out != obj) {
        CArray_INCREF(out);
        CArray_ResolveWritebackIfCopy(obj);
        CArray_DECREF(obj);
        obj = out;
    }

    if(out_ptr != NULL) {
        
        add_to_buffer(out_ptr, obj, sizeof(CArray));
    }
    CArrayDescriptor_FREE(indices_type);
    CArray_DECREF(target);
    CArray_DECREF(target);
    return obj;
fail:
    return NULL;
}