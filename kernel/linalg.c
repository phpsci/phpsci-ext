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
#include "lapacke.h"
#include "matlib.h"
#include "convert.h"

static float s_one;
static float s_zero;
static float s_minus_one;
static float s_ninf;
static float s_nan;
static double d_one;
static double d_zero;
static double d_minus_one;
static double d_ninf;
static double d_nan;

static inline void *
linearize_DOUBLE_matrix(double *dst_in,
                        double *src_in,
                        CArray * a)
{
    double *src = (double *) src_in;
    double *dst = (double *) dst_in;

    if (dst) {
        int i, j;
        double* rv = dst;
        int columns = (int)CArray_DIMS(a)[1];
        int column_strides = CArray_STRIDES(a)[1]/sizeof(double);
        int one = 1;
        for (i = 0; i < CArray_DIMS(a)[0]; i++) {
            if (column_strides > 0) {
                cblas_dcopy(columns,
                             (double*)src, column_strides,
                             (double*)dst, one);
            }
            else if (column_strides < 0) {
                cblas_dcopy(columns,
                             (double*)((double*)src + (columns-1)*column_strides),
                             column_strides,
                             (double*)dst, one);
            }
            else {
                /*
                 * Zero stride has undefined behavior in some BLAS
                 * implementations (e.g. OSX Accelerate), so do it
                 * manually
                 */
                for (j = 0; j < columns; ++j) {
                    memcpy((double*)dst + j, (double*)src, sizeof(double));
                }
            }

            src += CArray_STRIDES(a)[0]/sizeof(double);
            dst += CArray_DIMS(a)[1];
        }
        return rv;
    } else {
        return src;
    }
}

/**
 * DOT
 */
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


    if (CArray_NDIM(ap1) > 2 || CArray_NDIM(ap2) > 2) {
        throw_valueerror_exception("Matrix product is not implemented for DIM > 2");
        return NULL;
    }
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
    int status, casted = 0;
    int * ipiv = emalloc(sizeof(int) * CArray_DIMS(a)[0]);
    CArray * identity = CArray_Eye(CArray_DIMS(a)[0], CArray_DIMS(a)[0], 0, NULL, out);
    CArray * target;
    int order;
    double * data = emalloc(sizeof(double) * CArray_SIZE(a));

    if (CArray_NDIM(a) != 2) {
        throw_valueerror_exception("Matrix must have 2 dimensions");
        return NULL;
    }

    if (CArray_DESCR(a)->type_num != TYPE_DOUBLE_INT) {
        CArrayDescriptor *descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
        target = CArray_NewLikeArray(a, CARRAY_CORDER, descr, 0);
        if(CArray_CastTo(target, a) < 0) {
            return NULL;
        }
        casted = 1;
    } else {
        target = a;
    }

    if (!CArray_CHKFLAGS(a, CARRAY_ARRAY_C_CONTIGUOUS)) {
        linearize_DOUBLE_matrix(data, DDATA(target), a);
    } else {
        memcpy(data, DDATA(target), sizeof(double) * CArray_SIZE(target));
    }

    status = LAPACKE_dgesv(LAPACK_ROW_MAJOR,
            CArray_DIMS(target)[0],
            CArray_DIMS(target)[0],
            data,
            CArray_DIMS(target)[0],
            ipiv,
            DDATA(identity),
            CArray_DIMS(target)[0]);

    if (casted) {
        CArray_Free(target);
    }
    efree(ipiv);
    return identity;
}

/**
 *
 * @param a
 * @param norm 0 =  largest absolute value, 1 = Frobenius norm, 2 = infinity norm, 3 = 1-norm
 * @param out
 * @return
 */
CArray *
CArray_Norm(CArray * a, int norm, MemoryPointer * out)
{
    double result;
    char norm_c;
    CArray * target, * rtn;
    CArrayDescriptor * rtn_descr;
    int casted = 0;
    double * data;

    switch(norm) {
        case 0:
            norm_c = 'M';
            break;
        case 1:
            norm_c = 'F';
            break;
        case 2:
            norm_c = 'I';
            break;
        case 3:
            norm_c = '1';
            break;
        default:
            throw_valueerror_exception("Can't find a NORM algorithm with the provided name.");
            goto fail;
    }

    if (CArray_NDIM(a) != 2) {
        throw_valueerror_exception("Matrix must have 2 dimensions");
        goto fail;
    }

    if (CArray_DESCR(a)->type_num != TYPE_DOUBLE_INT) {
        CArrayDescriptor *descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
        target = CArray_NewLikeArray(a, CARRAY_CORDER, descr, 0);
        if(CArray_CastTo(target, a) < 0) {
            goto fail;
        }
        casted = 1;
    } else {
        target = a;
    }

    if (!CArray_CHKFLAGS(a, CARRAY_ARRAY_C_CONTIGUOUS)) {
        data = emalloc(sizeof(double) * CArray_SIZE(target));
        linearize_DOUBLE_matrix(data, DDATA(target), a);
    } else {
        data = DDATA(target);
    }


    rtn = emalloc(sizeof(CArray));
    rtn_descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
    rtn = CArray_NewFromDescr_int(rtn, rtn_descr, 0, NULL, NULL, NULL, 0, NULL, 0, 0);
    DDATA(rtn)[0] = LAPACKE_dlange(LAPACK_ROW_MAJOR,
            norm_c,
            CArray_DIMS(target)[0],
            CArray_DIMS(target)[1],
            data,
            CArray_DIMS(target)[0]);

    if (casted) {
        CArray_Free(target);
    }

    if(out != NULL) {
        add_to_buffer(out, rtn, sizeof(CArray));
    }
    return rtn;
fail:
    return NULL;
}

CArray *
CArray_Det(CArray * a, MemoryPointer * out)
{
    double result;
    int * ipiv = emalloc(sizeof(int) * CArray_DIMS(a)[0]);
    lapack_int status;
    double sign;
    int i;
    CArray * target, * rtn;
    CArrayDescriptor * rtn_descr;
    int casted = 0;
    double * data;

    if (CArray_NDIM(a) != 2) {
        throw_valueerror_exception("Expected matrix with 2 dimensions");
        goto fail;
    }

    if (CArray_DIMS(a)[0] != CArray_DIMS(a)[1]) {
        throw_valueerror_exception("Expected square matrix");
        goto fail;
    }
    if (CArray_DESCR(a)->type_num != TYPE_DOUBLE_INT) {
        CArrayDescriptor *descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
        target = CArray_NewLikeArray(a, CARRAY_CORDER, descr, 0);
        if(CArray_CastTo(target, a) < 0) {
            goto fail;
        }
        casted = 1;
    } else {
        target = a;
    }

    if (!CArray_CHKFLAGS(a, CARRAY_ARRAY_C_CONTIGUOUS)) {
        data = emalloc(sizeof(double) * CArray_SIZE(target));
        linearize_DOUBLE_matrix(data, DDATA(target), a);
    } else {
        data = DDATA(target);
    }


    status = LAPACKE_dgetrf(
            LAPACK_ROW_MAJOR,
            CArray_DIMS(a)[0],
            CArray_DIMS(a)[1],
            data,
            CArray_DIMS(a)[0],
            ipiv
            );

    int change_sign = 0;

    for (i = 0; i < CArray_DIMS(a)[0]; i++)
    {
        change_sign += (ipiv[i] != (i+1));
    }

    sign = (change_sign % 2)? -1.0 : 1.0;

    double acc_sign = sign;
    double acc_logdet = 0.0;
    double * src = data;

    for (i = 0; i < CArray_DIMS(a)[0]; i++) {
        double abs_element = *src;
        if (abs_element < 0.0) {
            acc_sign = -acc_sign;
            abs_element = -abs_element;
        }
        acc_logdet += log(abs_element);
        src += CArray_DIMS(a)[0]+1;
    }

    rtn = emalloc(sizeof(CArray));
    rtn_descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
    rtn = CArray_NewFromDescr_int(rtn, rtn_descr, 0, NULL, NULL, NULL, 0, NULL, 0, 0);

    DDATA(rtn)[0] = sign * exp(acc_logdet);

    if (casted) {
        CArray_Free(target);
    }

    if(out != NULL) {
        add_to_buffer(out, rtn, sizeof(CArray));
    }

    if (!CArray_CHKFLAGS(a, CARRAY_ARRAY_C_CONTIGUOUS)) {
        efree(data);
    }

    return rtn;
fail:
    return NULL;
}