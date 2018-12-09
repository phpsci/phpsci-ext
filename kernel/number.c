#include "number.h"
#include "carray.h"
#include "iterators.h"
#include "buffer.h"

void *
_carray_add_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    IDATA(out)[out_index] = *(IT_IDATA(a)) + *(IT_IDATA(b));
}

void *
_carray_add_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) + *(IT_DDATA(b));
}

/**
 * @param m1
 * @param m2
 * @return
 */
CArray *
CArray_Add(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else {
        prior1 = m2;
        prior2 = m1;
    }
    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        if(i < dim_diff) {
            dimensions[i] = CArray_DIM(prior1, i);
            continue;
        }
        if(CArray_DIM(prior1, i) < CArray_DIM(prior2, (i-dim_diff))) {
            dimensions[i] = CArray_DIM(prior2, i);
            continue;
        }
        dimensions[i] = CArray_DIM(prior1, i);
    }

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            data_op = &_carray_add_double;
            break;
        case TYPE_INTEGER_INT:
            data_op = &_carray_add_int;
            break;
    }

    result = CArray_NewFromDescr_int(result, CArray_DESCR(prior1), CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, CARRAY_NEEDS_INIT, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));

    i = 0;
    do {
        data_op(it1, it2, result, i);
        CArrayIterator_NEXT(it1);
        CArrayIterator_NEXT(it2);
        i++;
    } while(CArrayIterator_NOTDONE(it1));

    CArrayIterator_FREE(it1);
    CArrayIterator_FREE(it2);
    efree(dimensions);
    result->flags = (CARRAY_ARRAY_C_CONTIGUOUS | CARRAY_ARRAY_OWNDATA | CARRAY_ARRAY_WRITEABLE | CARRAY_ARRAY_ALIGNED);
    if(ptr != NULL) {
        add_to_buffer(ptr, result, sizeof(*result));
    }

    return result;
}