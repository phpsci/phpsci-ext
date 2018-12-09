#include "carray.h"
#include "common/exceptions.h"
#include "linalg.h"
#include "common/cblas_funcs.h"
#include "common/common.h"
#include "alloc.h"

/**
 * CArray Matmul
 **/ 
CArray * 
CArray_Matmul(CArray * ap1, CArray * ap2, CArray * out, MemoryPointer * ptr)
{
    CArray * out_buf = NULL, * result = NULL;
    int nd1, nd2, nd, typenum;
    int i, j, l, matchDim, is1, is2;
    int * dimensions;

    if (CArray_NDIM(ap1) == 0 || CArray_NDIM(ap2) == 0) {
        throw_valueerror_exception("Scalar operands are not allowed, use '*' instead");
        return NULL;
    }
    typenum = CArray_DESCR(ap1)->type_num;

    nd1 = CArray_NDIM(ap1);
    nd2 = CArray_NDIM(ap2);

    if (nd1 <= 2 && nd2 <= 2 && (TYPE_DOUBLE_INT == typenum || TYPE_FLOAT_INT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out, ptr);
    }

    l = CArray_DIMS(ap1)[CArray_NDIM(ap1) - 1];
    if (CArray_NDIM(ap2) > 1) {
        matchDim = CArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }
    if (CArray_DIMS(ap2)[matchDim] != l) {
        //dot_alignment_error(ap1, CArray_NDIM(ap1) - 1, ap2, matchDim);
        goto fail;
    }
    nd = CArray_NDIM(ap1) + CArray_NDIM(ap2) - 2;
    if (nd > CARRAY_MAXDIMS) {
        throw_valueerror_exception("dot: too many dimensions in result");
        goto fail;
    }

    j = 0;
    dimensions = emalloc((CArray_NDIM(ap1)) * sizeof(int));
    for (i = 0; i < CArray_NDIM(ap1) - 1; i++) {
        dimensions[j++] = CArray_DIMS(ap1)[i];
    }
    for (i = 0; i < CArray_NDIM(ap2) - 2; i++) {
        dimensions[j++] = CArray_DIMS(ap2)[i];
    }
    if(CArray_NDIM(ap2) > 1) {
        dimensions[j++] = CArray_DIMS(ap2)[CArray_NDIM(ap2)-1];
    }

    is1 = CArray_STRIDES(ap1)[CArray_NDIM(ap1)-1];
    is2 = CArray_STRIDES(ap2)[matchDim];
    /* Choose which subtype to return */
    out_buf = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);
    if (out_buf == NULL) {
        goto fail;
    }

    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (CArray_SIZE(ap1) == 0 && CArray_SIZE(ap2) == 0) {
        memset(CArray_DATA(out_buf), 0, CArray_NBYTES(out_buf));
    }
    php_printf("\n\n FOI \n\n");


fail:
    return NULL;

}