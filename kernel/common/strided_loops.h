#ifndef PHPSCI_EXT_STRIDE_LOOPS_H
#define PHPSCI_EXT_STRIDE_LOOPS_H

#include "../carray.h"

/*
 * This function pointer is for unary operations that input an
 * arbitrarily strided one-dimensional array segment and output
 * an arbitrarily strided array segment of the same size.
 * It may be a fully general function, or a specialized function
 * when the strides or item size have particular known values.
 *
 * Examples of unary operations are a straight copy, a byte-swap,
 * and a casting operation,
 *
 * The 'transferdata' parameter is slightly special, following a
 * generic auxiliary data pattern defined in carray.h
 * Use CARRAY_AUXDATA_CLONE and CARRAY_AUXDATA_FREE to deal with this data.
 *
 */
typedef void (CArray_StridedUnaryOp)(char *dst, int dst_stride,
                                     char *src, int src_stride,
                                     int N, int src_itemsize,
                                     CArrayAuxData *transferdata);


#endif //PHPSCI_EXT_STRIDE_LOOPS_H