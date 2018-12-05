#include "carray.h"
#include "common/exceptions.h"
#include "linalg.h"
#include "common/cblas_funcs.h"

/**
 * CArray Matmul
 **/ 
CArray * 
CArray_Matmul(CArray * ap1, CArray * ap2, CArray * out, MemoryPointer * ptr)
{
    int nd1, nd2, typenum;
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
    
}