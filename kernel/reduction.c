#include "carray.h"
#include "reduction.h"
#include "iterators.h"
#include "common/mem_overlap.h"
#include "scalar.h"
#include "convert.h"

/*
 * Allocates a result array for a reduction operation, with
 * dimensions matching 'arr' except set to 1 with 0 stride
 * wherever axis_flags is True. Dropping the reduction axes
 * from the result must be done later by the caller once the
 * computation is complete.
 *
 * This function always allocates a base class ndarray.
 *
 * If 'dtype' isn't NULL, this function steals its reference.
 */
static CArray *
allocate_reduce_result(CArray *arr, int *axis_flags,
                       CArrayDescriptor *dtype, int subok)
{
    int strides[CARRAY_MAXDIMS], stride;
    int shape[CARRAY_MAXDIMS], *arr_shape = CArray_DIMS(arr);
    ca_stride_sort_item strideperm[CARRAY_MAXDIMS];
    int idim, ndim = CArray_NDIM(arr);

    if (dtype == NULL) {
        dtype = CArray_DESCR(arr);
        CArrayDescriptor_INCREF(dtype);
    }

    CArray_CreateSortedStridePerm(CArray_NDIM(arr),
                                  CArray_STRIDES(arr), strideperm);

    /* Build the new strides and shape */
    stride = dtype->elsize;
    memcpy(shape, arr_shape, ndim * sizeof(shape[0]));
    for (idim = ndim-1; idim >= 0; --idim) {
        int i_perm = strideperm[idim].perm;
        if (axis_flags[i_perm]) {
            strides[i_perm] = 0;
            shape[i_perm] = 1;
        }
        else {
            strides[i_perm] = stride;
            stride *= shape[i_perm];
        }
    }

    /* Finally, allocate the array */
    return CArray_NewFromDescr(arr, dtype, ndim, shape, strides, NULL, 0, subok ? arr : NULL);
}

/*
 * Conforms an output parameter 'out' to have 'ndim' dimensions
 * with dimensions of size one added in the appropriate places
 * indicated by 'axis_flags'.
 *
 * The return value is a view into 'out'.
 */
static CArray *
conform_reduce_result(int ndim, int *axis_flags,
                      CArray *out, int keepdims, const char *funcname,
                      int need_copy)
{
    int strides[CARRAY_MAXDIMS], shape[CARRAY_MAXDIMS];
    int *strides_out = CArray_STRIDES(out);
    int *shape_out = CArray_DIMS(out);
    int idim, idim_out, ndim_out = CArray_NDIM(out);
    CArrayDescriptor *dtype;
    CArray *ret;

    /*
     * If the 'keepdims' parameter is true, do a simpler validation and
     * return a new reference to 'out'.
     */
    if (keepdims) {
        if (CArray_NDIM(out) != ndim) {
            throw_valueerror_exception("output parameter for reduction operation "
                                       "has the wrong number of dimensions (must match "
                                       "the operand's when keepdims=True)");
            return NULL;
        }

        for (idim = 0; idim < ndim; ++idim) {
            if (axis_flags[idim]) {
                if (shape_out[idim] != 1) {
                    throw_valueerror_exception(
                            "output parameter for reduction operation "
                            "has a reduction dimension not equal to one "
                            "(required when keepdims=True)");
                    return NULL;
                }
            }
        }

        CArray_INCREF(out);
        return out;
    }

    /* Construct the strides and shape */
    idim_out = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            strides[idim] = 0;
            shape[idim] = 1;
        }
        else {
            if (idim_out >= ndim_out) {
                throw_valueerror_exception(
                        "output parameter for reduction operation"
                        "does not have enough dimensions");
                return NULL;
            }
            strides[idim] = strides_out[idim_out];
            shape[idim] = shape_out[idim_out];
            ++idim_out;
        }
    }

    if (idim_out != ndim_out) {
        throw_valueerror_exception(
                "output parameter for reduction operation"
                "has too many dimensions");
        return NULL;
    }

    /* Allocate the view */
    dtype = CArray_DESCR(out);
    CArrayDescriptor_INCREF(dtype);

    /* TODO: use PyArray_NewFromDescrAndBase here once multiarray and umath
     *       are merged
     */
    ret =   CArray_NewFromDescr(
            ret, dtype,
            ndim, shape, strides, CArray_DATA(out),
            CArray_FLAGS(out), NULL);
    if (ret == NULL) {
        return NULL;
    }

    CArray_INCREF(out);
    if (CArray_SetBaseCArray(ret, out) < 0) {
        CArray_DECREF(ret);
        return NULL;
    }

    if (need_copy) {
        CArray *ret_copy;

        ret_copy = (CArray *)CArray_NewLikeArray(ret, CARRAY_ANYORDER, NULL, 0);
        if (ret_copy == NULL) {
            CArray_DECREF(ret);
            return NULL;
        }

        if (CArray_CopyInto(ret_copy, ret) != 0) {
            CArray_DECREF(ret);
            CArray_DECREF(ret_copy);
            return NULL;
        }

        if (CArray_SetWritebackIfCopyBase(ret_copy, ret) < 0) {
            CArray_DECREF(ret);
            CArray_DECREF(ret_copy);
            return NULL;
        }

        return ret_copy;
    }
    else {
        return ret;
    }
}


/*
 * Count the number of dimensions selected in 'axis_flags'
 */
static int
count_axes(int ndim, int *axis_flags)
{
    int idim;
    int naxes = 0;

    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            naxes++;
        }
    }
    return naxes;
}

CArray *
CArray_InitializeReduceResult(
                    CArray *result, CArray *operand,
                    int *axis_flags,
                    int *out_skip_first_count, const char *funcname)
{
    int *strides, *shape, shape_orig[CARRAY_MAXDIMS];
    CArray *op_view = NULL;
    int idim, ndim, nreduce_axes;

    ndim = CArray_NDIM(operand);

    /* Default to no skipping first-visit elements in the iteration */
    *out_skip_first_count = 0;

    /* Take a view into 'operand' which we can modify. */
    op_view = (CArray *)CArray_View(operand);
    if (op_view == NULL) {
        return NULL;
    }

    /*
     * Now copy the subarray of the first element along each reduction axis,
     * then return a view to the rest.
     *
     * Adjust the shape to only look at the first element along
     * any of the reduction axes. We count the number of reduction axes
     * at the same time.
     */
    shape = CArray_DIMS(op_view);
    nreduce_axes = 0;
    memcpy(shape_orig, shape, ndim * sizeof(int));
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim]) {
            if (shape[idim] == 0) {
                throw_valueerror_exception(
                             "zero-size array to reduction operation "
                             "which has no identity");
                CArray_DECREF(op_view);
                return NULL;
            }
            shape[idim] = 1;
            ++nreduce_axes;
        }
    }

    /*
     * Copy the elements into the result to start.
     */
    if (CArray_CopyInto(result, op_view) < 0) {
        CArray_DECREF(op_view);
        return NULL;
    }

    /*
     * If there is one reduction axis, adjust the view's
     * shape to only look at the remaining elements
     */
    if (nreduce_axes == 1) {
        strides = CArray_STRIDES(op_view);
        for (idim = 0; idim < ndim; ++idim) {
            if (axis_flags[idim]) {
                shape[idim] = shape_orig[idim] - 1;
                op_view->data += strides[idim];
            }
        }
    }
    /* If there are zero reduction axes, make the view empty */
    else if (nreduce_axes == 0) {
        for (idim = 0; idim < ndim; ++idim) {
            shape[idim] = 0;
        }
    }
    /*
     * Otherwise iterate over the whole operand, but tell the inner loop
     * to skip the elements we already copied by setting the skip_first_count.
     */
    else {
        *out_skip_first_count = CArray_SIZE(result);

        CArray_DECREF(op_view);
        CArray_INCREF(operand);
        op_view = operand;
    }

    return op_view;
}


CArray *
CArray_CreateReduceResult(CArray *operand, CArray *out,
                          CArrayDescriptor *dtype, int *axis_flags,
                          int keepdims, int subok,
                          const char *funcname)
{
    CArray *result;

    if (out == NULL) {
        /* This function steals the reference to 'dtype' */
        result = allocate_reduce_result(operand, axis_flags, dtype, subok);
    }
    else {
        int need_copy = 0;

        if (solve_may_share_memory(operand, out, 1) != 0) {
            need_copy = 1;
        }

        /* Steal the dtype reference */
        CArray_DECREF(dtype);
        result = conform_reduce_result(CArray_NDIM(operand), axis_flags,
                                       out, keepdims, funcname, need_copy);
    }

    return result;
}



CArray *
ReduceWrapper(CArray *operand, CArray *out,
                    CArray *wheremask,
                    CArrayDescriptor *operand_dtype,
                    CArrayDescriptor *result_dtype,
                    CARRAY_CASTING casting,
                    int *axis_flags, int reorderable,
                    int keepdims,
                    int subok,
                    CArrayScalar *identity,
                    CArray_ReduceLoopFunc *loop,
                    void *data, int buffersize, const char *funcname,
                    int errormask)
{
    CArray *result = NULL, *op_view = NULL;
    int skip_first_count = 0;

    /* Iterator parameters */
    CArrayIterator *iter = NULL;
    CArray *op[3];
    CArrayDescriptor *op_dtypes[3];
    uint32_t flags, op_flags[3];

    /* More than one axis means multiple orders are possible */
    if (!reorderable && count_axes(CArray_NDIM(operand), axis_flags) > 1) {
        throw_valueerror_exception("reduction operation is not reorderable so at most one axis may be specified");
        return NULL;
    }
    /* Can only use where with an initial ( from identity or argument) */
    if (wheremask != NULL && identity == NULL) {
        throw_valueerror_exception("reduction operation '%s' does not have an identity, so to use a where mask one has to specify 'initial'");
        return NULL;
    }

    /*
     * This either conforms 'out' to the ndim of 'operand', or allocates
     * a new array appropriate for this reduction.
     *
     * A new array with WRITEBACKIFCOPY is allocated if operand and out have memory
     * overlap.
     */
    CArrayDescriptor_INCREF(result_dtype);
    result = CArray_CreateReduceResult(operand, out,
                            result_dtype, axis_flags,
                            keepdims, subok, funcname);
    if (result == NULL) {
        goto fail;
    }

    /*
     * Initialize the result to the reduction unit if possible,
     * otherwise copy the initial values and get a view to the rest.
     */
    if (identity != NULL) {
        if (CArray_FillWithScalar(result, identity) < 0) {
            goto fail;
        }
        op_view = operand;
        CArray_INCREF(op_view);
    }
    else {
        op_view = CArray_InitializeReduceResult(
            result, operand, axis_flags, &skip_first_count, funcname);
        if (op_view == NULL) {
            goto fail;
        }
        /* empty op_view signals no reduction; but 0-d arrays cannot be empty */
        if ((CArray_SIZE(op_view) == 0) || (CArray_NDIM(operand) == 0)) {
            CArray_DECREF(op_view);
            op_view = NULL;
            goto finish;
        }
    }

    /* Set up the iterator */
    op[0] = result;
    op[1] = op_view;
    op_dtypes[0] = result_dtype;
    op_dtypes[1] = operand_dtype;

    flags = CARRAY_ITER_BUFFERED |
            CARRAY_ITER_EXTERNAL_LOOP |
            CARRAY_ITER_GROWINNER |
            CARRAY_ITER_DONT_NEGATE_STRIDES |
            CARRAY_ITER_ZEROSIZE_OK |
            CARRAY_ITER_REDUCE_OK |
            CARRAY_ITER_REFS_OK;
    op_flags[0] = CARRAY_ITER_READWRITE |
                  CARRAY_ITER_ALIGNED |
                  CARRAY_ITER_NO_SUBTYPE;
    op_flags[1] = CARRAY_ITER_READONLY |
                  CARRAY_ITER_ALIGNED;
    if (wheremask != NULL) {
        op[2] = wheremask;
        op_dtypes[2] = CArray_DescrFromType(TYPE_BOOL_INT);
        if (op_dtypes[2] == NULL) {
            goto fail;
        }
        op_flags[2] = CARRAY_ITER_READONLY;
    }

fail:
    return NULL;
finish:
    return NULL;    
}