#include "ctors.h"
#include "carray.h"
#include "alloc.h"
#include "convert.h"

int
setArrayFromSequence(CArray *a, CArray *s, int dim, int offset)
{
    int i, slen;
    int res = -1;

    /* INCREF on entry DECREF on exit */
    CArray_INCREF(s);

    if (dim > a->ndim) {
        throw_valueerror_exception("setArrayFromSequence: sequence/array dimensions mismatch.");
        goto fail;
    }

    slen = CArray_SIZE(s);
    if (slen != a->dimensions[dim]) {
        throw_valueerror_exception("setArrayFromSequence: sequence/array shape mismatch.");
        goto fail;
    }

    for (i = 0; i < slen; i++) {
        CArray *o = CArray_Slice_Index(s, i, NULL);
        if ((a->ndim - dim) > 1) {
            res = setArrayFromSequence(a, o, dim+1, offset);
        }
        else {
            res = a->descriptor->f->setitem(o->data, (a->data + offset), a);
        }
        CArray_DECREF(o);
        CArray_DECREF(s);
        CArrayDescriptor_FREE(o->descriptor);
        CArray_Free(o);
        if (res < 0) {
            goto fail;
        }
        offset += a->strides[dim];
    }
    CArray_DECREF(s);
    return 0;

fail:
    CArray_DECREF(s);
    return res;
}