#include "carray.h"
#include "buffer.h"

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
        return NULL;
    }

    if (check_and_adjust_axis_msg(&axis1, ndim) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, ndim) < 0) {
        return NULL;
    }

    if (axis1 == axis2) {
        throw_valueerror_exception("axis1 and axis2 cannot be the same");
        return NULL;
    }

    /* Get the shape and strides of the two axes */
    shape = CArray_SHAPE(self);
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

    if (ret == NULL) {
        return NULL;
    }

    if(rtn_ptr != NULL) {
        add_to_buffer(rtn_ptr, ret, sizeof(CArray));
    }

    return ret;
}