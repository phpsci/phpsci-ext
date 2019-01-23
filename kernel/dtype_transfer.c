#include "carray.h"
#include "dtype_transfer.h"
#include "common/strided_loops.h"
#include "assign.h"
#include "common/common.h"
#include "common/exceptions.h"
#include "php.h"
#include "descriptor.h"
#include "common/strided_loops.h"

/*************************** DEST SETZERO *******************************/

/* Sets dest to zero */
typedef struct {
    CArrayAuxData base;
    int dst_itemsize;
} _dst_memset_zero_data;

static void
_null_to_strided_memset_zero(char *dst,
                        int dst_stride,
                        char *CARRAY_UNUSED(src), int CARRAY_UNUSED(src_stride),
                        int N, int CARRAY_UNUSED(src_itemsize),
                        CArrayAuxData *data)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    int dst_itemsize = d->dst_itemsize;

    while (N > 0) {
        memset(dst, 0, dst_itemsize);
        dst += dst_stride;
        --N;
    }
}

/* zero-padded data copy function */
static CArrayAuxData *_dst_memset_zero_data_clone(CArrayAuxData *data)
{
    _dst_memset_zero_data *newdata =
            (_dst_memset_zero_data *)emalloc(
                                    sizeof(_dst_memset_zero_data));
    if (newdata == NULL) {
        return NULL;
    }

    memcpy(newdata, data, sizeof(_dst_memset_zero_data));

    return (CArrayAuxData *)newdata;
}

static void
_null_to_contig_memset_zero(char *dst,
                        int dst_stride,
                        char *CARRAY_UNUSED(src), int CARRAY_UNUSED(src_stride),
                        int N, int CARRAY_UNUSED(src_itemsize),
                        CArrayAuxData *data)
{
    _dst_memset_zero_data *d = (_dst_memset_zero_data *)data;
    int dst_itemsize = d->dst_itemsize;
    memset(dst, 0, N*dst_itemsize);
}

static void
_dec_src_ref_nop(char *CARRAY_UNUSED(dst),
                        int CARRAY_UNUSED(dst_stride),
                        char *CARRAY_UNUSED(src), int CARRAY_UNUSED(src_stride),
                        int CARRAY_UNUSED(N),
                        int CARRAY_UNUSED(src_itemsize),
                        CArrayAuxData *CARRAY_UNUSED(data))
{
    /* NOP */
}

/*
 * Returns a transfer function which zeros out the dest values.
 *
 * Returns CARRAY_SUCCEED or CARRAY_FAIL.
 */
static int
get_setdstzero_transfer_function(int aligned,
                            int dst_stride,
                            CArrayDescriptor *dst_dtype,
                            CArray_StridedUnaryOp **out_stransfer,
                            CArrayAuxData **out_transferdata,
                            int *out_needs_api);

/*
 * Returns a transfer function which DECREFs any references in src_type.
 *
 * Returns CARRAY_SUCCEED or CARRAY_FAIL.
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

    src_itemsize = src_dtype->elsize;
    dst_itemsize = dst_dtype->elsize;
    src_type_num = src_dtype->type_num;
    dst_type_num = dst_dtype->type_num;
    is_builtin = src_type_num < CARRAY_NTYPES && dst_type_num < CARRAY_NTYPES;

    /*
     * If there are no references and the data types are equivalent and builtin,
     * return a simple copy
     */
    if (CArray_EquivTypes(src_dtype, dst_dtype) &&
            !CArrayDataType_REFCHK(src_dtype) && !CArrayDataType_REFCHK(dst_dtype) &&
            is_builtin) {
        /*
         * We can't pass through the aligned flag because it's not
         * appropriate. Consider a size-8 string, it will say it's
         * aligned because strings only need alignment 1, but the
         * copy function wants to know if it's alignment 8.
         *
         * TODO: Change align from a flag to a "best power of 2 alignment"
         *       which holds the strongest alignment value for all
         *       the data which will be used.
         */
        *out_stransfer = CArray_GetStridedCopyFn(0,
                                        src_stride, dst_stride,
                                        src_dtype->elsize);
        *out_transferdata = NULL;
        return CARRAY_SUCCEED;
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
     /* If there are no references, it's a nop */
    if (!CArrayDataType_REFCHK(src_dtype)) {
        *out_stransfer = &_dec_src_ref_nop;
        *out_transferdata = NULL;

        return CARRAY_SUCCEED;
    }
    throw_notimplemented_exception();
}                                                               

int
get_setdstzero_transfer_function(int aligned,
                                 int dst_stride,
                                 CArrayDescriptor *dst_dtype,
                                 CArray_StridedUnaryOp **out_stransfer,
                                 CArrayAuxData **out_transferdata,
                                 int *out_needs_api)
{
    _dst_memset_zero_data *data;
    /* If there are no references, just set the whole thing to zero */
    if (!CArrayDataType_REFCHK(dst_dtype)) {
        data = (_dst_memset_zero_data *)
                        emalloc(sizeof(_dst_memset_zero_data));
        if (data == NULL) {
            throw_memory_exception("Memory Error");
            return CARRAY_FAIL;
        }

        data->base.free = (CArrayAuxData_FreeFunc *)(&free);
        data->base.clone = &_dst_memset_zero_data_clone;
        data->dst_itemsize = dst_dtype->elsize;

        if (dst_stride == data->dst_itemsize) {
            *out_stransfer = &_null_to_contig_memset_zero;
        }
        else {
            *out_stransfer = &_null_to_strided_memset_zero;
        }
        *out_transferdata = (CArrayAuxData *)data;

        return CARRAY_SUCCEED;
    }
    throw_notimplemented_exception();
}
