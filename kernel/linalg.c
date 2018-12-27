#include "iterators.h"
#include "carray.h"
#include "common/exceptions.h"
#include "linalg.h"
#include "common/cblas_funcs.h"
#include "common/common.h"
#include "convert_type.h"
#include "alloc.h"
#include "buffer.h"
#include "cblas.h"

void
FLOAT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n)
{
}

void
DOUBLE_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n)
{
    int is1b = blas_stride(is1, sizeof(double));
    int is2b = blas_stride(is2, sizeof(double));

    if (is1b && is2b)
    {
        double sum = 0.;
        while (n > 0) {
            int chunk = n < CARRAY_CBLAS_CHUNK ? n : CARRAY_CBLAS_CHUNK;

            sum += cblas_ddot(chunk,
                    (double *) ip1, is1b,
            (double *) ip2, is2b);
            /* use char strides here */
            ip1 += chunk * is1;
            ip2 += chunk * is2;
            n -= chunk;
        }
        *((double *)op) = (double)sum;
    }
    else {
        double sum = (double)0;
        int i;
        for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
            const double ip1r = *((double *)ip1);
            const double ip2r = *((double *)ip2);
            sum += ip1r * ip2r;
        }
        *((double *)op) = sum;
    }
}

void
INT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n)
{
    int tmp = (int)0;
    int i;

    for (i = 0; i < n; i++, ip1 += is1, ip2 += is2) {
        tmp += (int)(*((int *)ip1)) *
        (int)(*((int *)ip2));
    }
    *((int *)op) = (int) tmp;
}

/**
 * CArray Matmul
 **/ 
CArray * 
CArray_Matmul(CArray * ap1, CArray * ap2, CArray * out, MemoryPointer * ptr)
{
    CArray * result = NULL;
    int nd1, nd2, nd, typenum;
    int i, j, l, matchDim, is1, is2, axis, os;
    int * dimensions;
    CArray_DotFunc *dot;
    CArrayIterator * it1, * it2;
    char * op;

    if (CArray_NDIM(ap1) == 0 || CArray_NDIM(ap2) == 0) {
        throw_valueerror_exception("Scalar operands are not allowed, use '*' instead");
        return NULL;
    }
    typenum = CArray_ObjectType(ap1, 0);
    typenum = CArray_ObjectType(ap2, typenum);

    nd1 = CArray_NDIM(ap1);
    nd2 = CArray_NDIM(ap2);

    if (nd1 <= 2 && nd2 <= 2 && (TYPE_DOUBLE_INT == typenum || TYPE_FLOAT_INT == typenum)) {
        return cblas_matrixproduct(typenum, ap1, ap2, out, ptr);
    }

    if(typenum == TYPE_INTEGER_INT) {
        dot = &INT_dot;
    }
    if(typenum == TYPE_FLOAT_INT) {
        dot = &FLOAT_dot;
    }
    if(typenum == TYPE_DOUBLE_INT) {
        dot = &DOUBLE_dot;
    }

    l = CArray_DIMS(ap1)[CArray_NDIM(ap1) - 1];
    if (CArray_NDIM(ap2) > 1) {
        matchDim = CArray_NDIM(ap2) - 2;
    }
    else {
        matchDim = 0;
    }

    if (CArray_DIMS(ap2)[matchDim] != l) {
        throw_valueerror_exception("Shapes are not aligned.");
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
    result = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);

    if (result == NULL) {
        goto fail;
    }

    /* Ensure that multiarray.dot(<Nx0>,<0xM>) -> zeros((N,M)) */
    if (CArray_SIZE(ap1) == 0 && CArray_SIZE(ap2) == 0) {
        memset(CArray_DATA(result), 0, CArray_NBYTES(result));
    }

    op = CArray_DATA(result);
    os = CArray_DESCR(result)->elsize;
    axis = CArray_NDIM(ap1)-1;
    it1 = CArray_IterAllButAxis(ap1, &axis);
    if (it1 == NULL) {
        goto fail;
    }

    it2 = CArray_IterAllButAxis(ap2, &matchDim);
    if (it2 == NULL) {
        goto fail;
    }

    while (it1->index < it1->size) {
        while (it2->index < it2->size) {
            dot(it1->data_pointer, is1, it2->data_pointer, is2, op, l);
            op += os;
            CArrayIterator_NEXT(it2);
        }
        CArrayIterator_NEXT(it1);
        CArrayIterator_RESET(it2);
    }
    CArrayIterator_FREE(it1);
    CArrayIterator_FREE(it2);

    if(ptr != NULL) {
        add_to_buffer(ptr, result, sizeof(CArray*));
    }
    efree(dimensions);
    // Remove appended dimension
    result->ndim = ap1->ndim;

    return result;
fail:
    if(dimensions != NULL) {
        efree(dimensions);
    }
    return NULL;
}

/**
 * Compute matrix inverse
 **/ 
CArray *
CArray_Inv(CArray * a, MemoryPointer * out) {
    
}