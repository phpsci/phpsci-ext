#include "carray.h"
#include "dtype_transfer.h"
#include "common/strided_loops.h"
#include "assign.h"
#include "common/common.h"
#include "common/exceptions.h"

/*
 * Returns a transfer function which DECREFs any references in src_type.
 *
 * Returns NPY_SUCCEED or NPY_FAIL.
 */
static int
get_decsrcref_transfer_function(int aligned,
                                int src_stride,
                                CArrayDescriptor *src_dtype,
                                CArray_StridedUnaryOp **out_stransfer,
                                CArrayAuxData **out_transferdata,
                                int *out_needs_api);


int 
CArray_CastRawArrays(int count, char *src, char *dst,
                     int src_stride, int dst_stride,
                     CArrayDescriptor *src_dtype, CArrayDescriptor *dst_dtype,
                     int move_references)
{
    CArray_StridedUnaryOp *stransfer = NULL;
    CArrayAuxData *transferdata = NULL;
    int aligned = 1, needs_api = 0;


    /* Make sure the copy is reasonable */
    if (dst_stride == 0 && count > 1) {
        throw_valueerror_exception("CArray CastRawArrays cannot do a reduction");
        return CARRAY_FAIL;
    }
    else if (count == 0) {
        return CARRAY_SUCCEED;
    }

    /* Check data alignment, both uint and true */
    aligned = raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   carray_uint_alignment(dst_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, dst, &dst_stride,
                                   dst_dtype->alignment) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   carray_uint_alignment(src_dtype->elsize)) &&
              raw_array_is_aligned(1, &count, src, &src_stride,
                                   src_dtype->alignment);

    /* Get the function to do the casting */
    if (CArray_GetDTypeTransferFunction(aligned,
                        src_stride, dst_stride,
                        src_dtype, dst_dtype,
                        move_references,
                        &stransfer, &transferdata,
                        &needs_api) != CARRAY_SUCCEED) {
        return CARRAY_FAIL;
    }                               
}


/********************* MAIN DTYPE TRANSFER FUNCTION ***********************/
int
CArray_GetDTypeTransferFunction(int aligned,
                            int src_stride, int dst_stride,
                            CArrayDescriptor *src_dtype, CArrayDescriptor *dst_dtype,
                            int move_references,
                            CArray_StridedUnaryOp **out_stransfer,
                            CArrayAuxData **out_transferdata,
                            int *out_needs_api)
{
    int src_itemsize, dst_itemsize;
    int src_type_num, dst_type_num;
    int is_builtin;
    
    /*
     * If one of the dtypes is NULL, we give back either a src decref
     * function or a dst setzero function
     */
    if (dst_dtype == NULL) {
        if (move_references) {
            return get_decsrcref_transfer_function(aligned,
                                src_dtype->elsize,
                                src_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
        }
        else {
            *out_stransfer = &_dec_src_ref_nop;
            *out_transferdata = NULL;
            return CARRAY_SUCCEED;
        }
    }
    else if (src_dtype == NULL) {
        return get_setdstzero_transfer_function(aligned,
                                dst_dtype->elsize,
                                dst_dtype,
                                out_stransfer, out_transferdata,
                                out_needs_api);
    }
}


int
get_decsrcref_transfer_function(int aligned,
                                int src_stride,
                                CArrayDescriptor *src_dtype,
                                CArray_StridedUnaryOp **out_stransfer,
                                CArrayAuxData **out_transferdata,
                                int *out_needs_api)
{
    
}                                                               