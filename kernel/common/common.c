#include "../carray.h"
#include "common.h"
#include "exceptions.h"

/*
 * Make a new empty array, of the passed size, of a type that takes the
 * priority of ap1 and ap2 into account.
 *
 * If `out` is non-NULL, memory overlap is checked with ap1 and ap2, and an
 * updateifcopy temporary array may be returned. If `result` is non-NULL, the
 * output array to be returned (`out` if non-NULL and the newly allocated array
 * otherwise) is incref'd and put to *result.
 */
CArray * 
new_array_for_sum(CArray *ap1, CArray *ap2, CArray* out,
                  int nd, int dimensions[], int typenum, CArray **result)
{
    CArray *out_buf;

    if (out) {
        int d;

        /* verify that out is usable */
        if (CArray_NDIM(out) != nd ||
            CArray_TYPE(out) != typenum) {
            throw_valueerror_exception(
                "output array is not acceptable (must have the right datatype, "
                "number of dimensions, and be a C-Array)");
            return 0;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != CArray_DIM(out, d)) {
                throw_valueerror_exception(
                    "output array has wrong dimensions");
                return 0;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = CArray_NewLikeArray(out, CARRAY_CORDER, NULL, 0);
            if (out_buf == NULL) {
                return NULL;
            }

            /* set copy-back */
            CArray_INCREF(out);
            if (CArray_SetWritebackIfCopyBase(out_buf, out) < 0) {
                CArray_DECREF(out);
                CArray_DECREF(out_buf);
                return NULL;
            }
        }
        else {
            CArray_INCREF(out);
            out_buf = out;
        }

        if (result) {
            CArray_INCREF(out);
            *result = out;
        }

        return out_buf;
    }
    else {
        CArray * subtype;
        double prior1, prior2;

        prior1 = prior2 = 0.0;
        subtype = ap1;
        
        out_buf = CArray_New(subtype, nd, dimensions,
                             typenum, NULL, NULL, 0, 0, 
                             (prior2 > prior1 ? ap2 : ap1));
        return out_buf;
    }
}                  