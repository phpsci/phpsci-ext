/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 * 
 * Original File
 * Copyright (c) NumPy (/numpy/core/src/common/cblasfuncs.c)
 * 
 * Modified for CArrays in 2018
 * 
 * Henrique Borba
 * henrique.borba.dev@gmail.com
 */

#include "../carray.h"
#include "../convert.h"
#include "cblas_funcs.h"
#include "common.h"
#include "cblas.h"

/*
 * Helper: dispatch to appropriate cblas_?gemm for typenum.
 */
static void
gemm(int typenum, enum CBLAS_ORDER order,
     enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     CArray *A, int lda, CArray *B, int ldb, CArray *R)
{
    int i ;
    const void *Adata = CArray_DATA(A), *Bdata = CArray_DATA(B);
    void *Rdata = CArray_DATA(R);
    int ldc = CArray_DIM(R, 1) > 1 ? CArray_DIM(R, 1) : 1;
   
    
    switch (typenum) {
        case TYPE_DOUBLE_INT:
            cblas_dgemm(order, transA, transB, m, n, k, 1.,
                        Adata, lda, Bdata, ldb, 0., Rdata, ldc);
            break;
        case TYPE_FLOAT_INT:
            cblas_sgemm(order, transA, transB, m, n, k, 1.f,
                        Adata, lda, Bdata, ldb, 0.f, Rdata, ldc);
            break;
    }    

}

/*
 * Helper: dispatch to appropriate cblas_?syrk for typenum.
 */
static void
syrk(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     int n, int k,
     CArray *A, int lda, CArray *R)
{
    const void *Adata = CArray_DATA(A);
    void *Rdata = CArray_DATA(R);
    int ldc = CArray_DIM(R, 1) > 1 ? CArray_DIM(R, 1) : 1;

    int i;
    int j;

    switch (typenum) {
        case TYPE_DOUBLE_INT:
            cblas_dsyrk(order, CblasUpper, trans, n, k, 1.,
                        Adata, lda, 0., Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((double*)CArray_GETPTR2(R, j, i)) =
                            *((double*)CArray_GETPTR2(R, i, j));
                }
            }
            break;
        case TYPE_FLOAT_INT:
            cblas_ssyrk(order, CblasUpper, trans, n, k, 1.f,
                        Adata, lda, 0.f, Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((float*)CArray_GETPTR2(R, j, i)) =
                            *((float*)CArray_GETPTR2(R, i, j));
                }
            }
            break;
    }
}

static MatrixShape
_select_matrix_shape(CArray *array)
{
    switch (CArray_NDIM(array)) {
        case 0:
            return _scalar;
        case 1:
            if (CArray_DIM(array, 0) > 1)
                return _column;
            return _scalar;
        case 2:
            if (CArray_DIM(array, 0) > 1) {
                if (CArray_DIM(array, 1) == 1)
                    return _column;
                else
                    return _matrix;
            }
            if (CArray_DIM(array, 1) == 1)
                return _scalar;
            return _row;
    }
    return _matrix;
}

/*
 * This also makes sure that the data segment is aligned with
 * an itemsize address as well by returning one if not true.
 */
static int
_bad_strides(CArray * ap)
{
    int itemsize = CArray_ITEMSIZE(ap);
    int i, N=CArray_NDIM(ap);
    int *strides = CArray_STRIDES(ap);

    if ((*((int *)(CArray_DATA(ap))) % itemsize) != 0) {
        return 1;
    }
    for (i = 0; i < N; i++) {
        if ((strides[i] < 0) || (strides[i] % itemsize) != 0) {
            return 1;
        }
    }

    return 0;
}

/**
 * Dot product of ap1 and ap2 using CBLAS.
 */ 
CArray * 
cblas_matrixproduct(int typenum, CArray * ap1, CArray *ap2, CArray *out, MemoryPointer * ptr)
{
    CArray * result = NULL, * out_buffer = NULL;
    int j, lda, ldb;

    int nd, l;
    int ap1stride = 0;
    int * dimensions = NULL;
    int numbytes;
    MatrixShape ap1shape, ap2shape;
    out_buffer = (CArray *)emalloc(sizeof(CArray));

    if(_bad_strides(ap1)) {
        CArray * op1 = CArray_NewCopy(ap1, CARRAY_ANYORDER);
        CArray_DECREF(ap1);
        ap1 = op1;
        if(ap1 == NULL) {
            goto fail;
        }
    }
    if (_bad_strides(ap2)) {
        CArray * op2 = CArray_NewCopy(ap2, CARRAY_ANYORDER);
        CArray_DECREF(ap2);
        ap2 = op2;
        if (ap2 == NULL) {
            goto fail;
        }
    }
    
    ap1shape = _select_matrix_shape(ap1);
    ap2shape = _select_matrix_shape(ap2);

    if (ap1shape == _scalar || ap2shape == _scalar) {
        CArray *oap1, *oap2;
        oap1 = ap1; oap2 = ap2;
        /* One of ap1 or ap2 is a scalar */
        if (ap1shape == _scalar) {
            /* Make ap2 the scalar */
            CArray *t = ap1;
            ap1 = ap2;
            ap2 = t;
            ap1shape = ap2shape;
            ap2shape = _scalar;
        }
        if (ap1shape == _row) {
            ap1stride = CArray_STRIDE(ap1, 1);
        }
        else if (CArray_NDIM(ap1) > 0) {
            ap1stride = CArray_STRIDE(ap1, 0);
        }

        if (CArray_NDIM(ap1) == 0 || CArray_NDIM(ap2) == 0) {
            int *thisdims;
            if (CArray_NDIM(ap1) == 0) {
                nd = CArray_NDIM(ap2);
                thisdims = CArray_DIMS(ap2);
            }
            else {
                nd = CArray_NDIM(ap1);
                thisdims = CArray_DIMS(ap1);
            }
            l = 1;
            dimensions = (int*)emalloc(nd * sizeof(int));
            for (j = 0; j < nd; j++) {
                dimensions[j] = thisdims[j];
                l *= dimensions[j];
            }
        } 
        else {
            l = CArray_DIM(oap1, CArray_NDIM(oap1) - 1);

            if (CArray_DIM(oap2, 0) != l) {
                goto fail;
            }
            nd = CArray_NDIM(ap1) + CArray_NDIM(ap2) - 2;
            dimensions = (int*)emalloc(nd * sizeof(int));
            /*
             * nd = 0 or 1 or 2. If nd == 0 do nothing ...
             */
            if (nd == 1) {
                /*
                 * Either CArray_DIM(ap1) is 1 dim or CArray_DIM(ap2) is
                 * 1 dim and the other is 2 dim
                 */
                dimensions[0] = (CArray_NDIM(oap1) == 2) ?
                                CArray_DIM(oap1, 0) : CArray_DIM(oap2, 1);
                l = dimensions[0];
                /*
                 * Fix it so that dot(shape=(N,1), shape=(1,))
                 * and dot(shape=(1,), shape=(1,N)) both return
                 * an (N,) array (but use the fast scalar code)
                 */
            }
            else if (nd == 2) {
                dimensions[0] = CArray_DIM(oap1, 0);
                dimensions[1] = CArray_DIM(oap2, 1);
                /*
                 * We need to make sure that dot(shape=(1,1), shape=(1,N))
                 * and dot(shape=(N,1),shape=(1,1)) uses
                 * scalar multiplication appropriately
                 */
                if (ap1shape == _row) {
                    l = dimensions[1];
                }
                else {
                    l = dimensions[0];
                }
            }

            /* Check if the summation dimension is 0-sized */
            if (CArray_DIM(oap1, CArray_NDIM(oap1) - 1) == 0) {
                l = 0;
            }
        }
    } else {
        /*
         * (CArray_NDIM(ap1) <= 2 && CArray_NDIM(ap2) <= 2)
         * Both ap1 and ap2 are vectors or matrices
         */
        l = CArray_DIM(ap1, CArray_NDIM(ap1) - 1);

        if (CArray_DIM(ap2, 0) != l) {
            dot_alignment_error(ap1, CArray_NDIM(ap1) - 1, ap2, 0);
            goto fail;
        }
        nd = CArray_NDIM(ap1) + CArray_NDIM(ap2) - 2;
        dimensions = (int*)emalloc(nd * sizeof(int));
        if (nd == 1) {
            dimensions[0] = (CArray_NDIM(ap1) == 2) ?
                            CArray_DIM(ap1, 0) : CArray_DIM(ap2, 1);
        }
        else if (nd == 2) {
            dimensions[0] = CArray_DIM(ap1, 0);
            dimensions[1] = CArray_DIM(ap2, 1);
        }
    }
    
    out_buffer = new_array_for_sum(ap1, ap2, out, nd, dimensions, typenum, &result);

    if (out_buffer == NULL) {
        goto fail;
    }
    
    numbytes = CArray_NBYTES(out_buffer);
    
    if (numbytes == 0 || l == 0) {
            CArray_DECREF(ap1);
            CArray_DECREF(ap2);
            CArray_DECREF(out_buffer);
            return result;
    }

    if (ap2shape == _scalar) {
        /*
         * Multiplication by a scalar -- Level 1 BLAS
         * if ap1shape is a matrix and we are not contiguous, then we can't
         * just blast through the entire array using a single striding factor
         */
        
        if (typenum == TYPE_DOUBLE) {

        }
    } else if ((ap2shape == _column) && (ap1shape != _matrix)) {
        
    } else if (ap1shape == _matrix && ap2shape != _matrix) {
        
    } else if (ap1shape != _matrix && ap2shape == _matrix) {
        
    } else {
        
        /*
         * (CArray_NDIM(ap1) == 2 && CArray_NDIM(ap2) == 2)
         * Matrix matrix multiplication -- Level 3 BLAS
         *  L x M  multiplied by M x N
         */
        enum CBLAS_ORDER Order;
        enum CBLAS_TRANSPOSE Trans1, Trans2;
        int M, N, L;

        /* Optimization possible: */
        /*
         * We may be able to handle single-segment arrays here
         * using appropriate values of Order, Trans1, and Trans2.
         */
        if (!CArray_IS_C_CONTIGUOUS(ap2) && !CArray_IS_F_CONTIGUOUS(ap2)) {
            CArray *new = CArray_Copy(ap2);

            CArray_DECREF(ap2);
            ap2 = new;
            if (new == NULL) {
                goto fail;
            }
        }
        if (!CArray_IS_C_CONTIGUOUS(ap1) && !CArray_IS_F_CONTIGUOUS(ap1)) {
            
            CArray *new = CArray_Copy(ap1);
            CArray_DECREF(ap1);
            ap1 = new;
            if (new == NULL) {
                goto fail;
            }
        }

        Order = CblasRowMajor;
        Trans1 = CblasNoTrans;
        Trans2 = CblasNoTrans;
        L = CArray_DIM(ap1, 0);
        N = CArray_DIM(ap2, 1);
        M = CArray_DIM(ap2, 0);
        lda = (CArray_DIM(ap1, 1) > 1 ? CArray_DIM(ap1, 1) : 1);
        ldb = (CArray_DIM(ap2, 1) > 1 ? CArray_DIM(ap2, 1) : 1);

        /*
         * Avoid temporary copies for arrays in Fortran order
         */
        if (CArray_IS_F_CONTIGUOUS(ap1)) {
            Trans1 = CblasTrans;
            lda = (CArray_DIM(ap1, 0) > 1 ? CArray_DIM(ap1, 0) : 1);
        }
        if (CArray_IS_F_CONTIGUOUS(ap2)) {
            Trans2 = CblasTrans;
            ldb = (CArray_DIM(ap2, 0) > 1 ? CArray_DIM(ap2, 0) : 1);
        }

        /*
         * Use syrk if we have a case of a matrix times its transpose.
         * Otherwise, use gemm for all other cases.
         */
        if (
            (CArray_BYTES(ap1) == CArray_BYTES(ap2)) &&
            (CArray_DIM(ap1, 0) == CArray_DIM(ap2, 1)) &&
            (CArray_DIM(ap1, 1) == CArray_DIM(ap2, 0)) &&
            (CArray_STRIDE(ap1, 0) == CArray_STRIDE(ap2, 1)) &&
            (CArray_STRIDE(ap1, 1) == CArray_STRIDE(ap2, 0)) &&
            ((Trans1 == CblasTrans) ^ (Trans2 == CblasTrans)) &&
            ((Trans1 == CblasNoTrans) ^ (Trans2 == CblasNoTrans))
        ) {
            if (Trans1 == CblasNoTrans) {
                syrk(typenum, Order, Trans1, N, M, ap1, lda, out_buffer);
            }
            else {
                syrk(typenum, Order, Trans1, N, M, ap2, ldb, out_buffer);
            }
        }
        else {
            gemm(typenum, Order, Trans1, Trans2, L, N, M, ap1, lda, ap2, ldb,
                 out_buffer);
            CArray_Print(out_buffer);
        }
    }
    
    CArray_DECREF(ap1);
    CArray_DECREF(ap2);

    /* Trigger possible copyback into `result` */
    CArray_ResolveWritebackIfCopy(out_buffer);
    CArray_DECREF(out_buffer);
    
    if(dimensions != NULL) {
        efree(dimensions);
    }

    if(ptr != NULL ){
        add_to_buffer(ptr, *(out_buffer), sizeof(CArray));
    }

    return out_buffer;
fail:
    efree(dimensions);
    return NULL;    
}