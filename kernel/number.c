#include "number.h"
#include "carray.h"
#include "iterators.h"
#include "buffer.h"
#include "convert_type.h"

void *
_carray_add_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) + *(IT_DDATA(b));
}

void *
_carray_add_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) + *(IT_IDATA(b));
}

void *
_carray_add_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_IDATA(a)) + *(IT_DDATA(b));
}

void *
_carray_add_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    IDATA(out)[out_index] = *(IT_IDATA(a)) + *(IT_IDATA(b));
}

void *
_carray_sub_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) - *(IT_DDATA(b));
}

void *
_carray_sub_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) - *(IT_IDATA(b));
}

void *
_carray_sub_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_IDATA(a)) - *(IT_DDATA(b));
}

void *
_carray_sub_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    IDATA(out)[out_index] = *(IT_IDATA(a)) - *(IT_IDATA(b));
}

void *
_carray_mult_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = (double)*(IT_DDATA(a)) * (double)*(IT_DDATA(b));
}

void *
_carray_mult_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) * (double)*(IT_IDATA(b));
}

void *
_carray_mult_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_IDATA(a)) * *(IT_DDATA(b));
}

void *
_carray_mult_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    IDATA(out)[out_index] = *(IT_IDATA(a)) * *(IT_IDATA(b));
}

void *
_carray_divide_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) / *(IT_DDATA(b));
}

void *
_carray_divide_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = *(IT_DDATA(a)) / (double)*(IT_IDATA(b));
}

void *
_carray_divide_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = (double)*(IT_IDATA(a)) / *(IT_DDATA(b));
}

void *
_carray_divide_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = (double)((double)*(IT_IDATA(a)) / (double)*(IT_IDATA(b)));
}

void *
_carray_power_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = pow(*(IT_DDATA(a)), *(IT_DDATA(b)));
}

void *
_carray_power_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = pow(*(IT_DDATA(a)), *(IT_IDATA(b)));
}

void *
_carray_power_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = pow(*(IT_IDATA(a)), *(IT_DDATA(b)));
}

void *
_carray_power_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    IDATA(out)[out_index] = pow(*(IT_IDATA(a)) , *(IT_IDATA(b)));
}

void *
_carray_mod_double_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = fmod(*(IT_DDATA(a)), *(IT_DDATA(b)));
}

void *
_carray_mod_double_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = fmod(*(IT_DDATA(a)), (double)*(IT_IDATA(b)));
}

void *
_carray_mod_int_double(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = fmod((double)*(IT_IDATA(a)), *(IT_DDATA(b)));
}

void *
_carray_mod_int_int(CArrayIterator * a, CArrayIterator * b, CArray * out, int out_index) {
    DDATA(out)[out_index] = (double)fmod((double)*(IT_IDATA(a)), (double)*(IT_IDATA(b)));
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
    int typenum;
    CArrayDescriptor * type;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    typenum = CArray_ObjectType(prior1, 0.0);
    typenum = CArray_ObjectType(prior2, typenum);
    type = CArray_DescrFromType(typenum);

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_add_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_add_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_add_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_add_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));

    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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

CArray *
CArray_Subtract(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;
    int typenum;
    CArrayDescriptor * type;
    int inverted = 0;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
            inverted = 1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
        inverted = 1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    typenum = CArray_ObjectType(prior1, 0.0);
    typenum = CArray_ObjectType(prior2, typenum);
    type = CArray_DescrFromType(typenum);

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_sub_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_sub_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_sub_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_sub_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));
    CArrayIterator * tmp_it;

    if (inverted) {
        tmp_it = it1;
        it1 = it2;
        it2 = tmp_it;
    }


    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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

CArray *
CArray_Multiply(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;
    int typenum;
    CArrayDescriptor * type;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    typenum = CArray_ObjectType(prior1, 0.0);
    typenum = CArray_ObjectType(prior2, typenum);
    type = CArray_DescrFromType(typenum);

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_mult_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_mult_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_mult_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_mult_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));

    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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

CArray *
CArray_Divide(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;
    int typenum, inverted = 0;
    CArrayDescriptor * type;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
            inverted = 1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
        inverted = 1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    type = CArray_DescrFromType(TYPE_DOUBLE_INT);

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_divide_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_divide_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_divide_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_divide_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));
    CArrayIterator * tmp_it;

    if (inverted) {
        tmp_it = it1;
        it1 = it2;
        it2 = tmp_it;
    }

    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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
CArray *
CArray_Power(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;
    int typenum;
    CArrayDescriptor * type;
    int inverted = 0;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
            inverted = 1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
        inverted = 1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    typenum = CArray_ObjectType(prior1, 0.0);
    typenum = CArray_ObjectType(prior2, typenum);
    type = CArray_DescrFromType(typenum);


    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_power_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_power_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_power_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_power_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));
    CArrayIterator * tmp_it;
    if (inverted) {
        tmp_it = it1;
        it1 = it2;
        it2 = tmp_it;
    }


    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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

CArray *
CArray_Mod(CArray *m1, CArray *m2, MemoryPointer * ptr)
{
    CArray * prior1, * prior2, * result;
    void * (*data_op)(CArrayIterator *, CArrayIterator *, CArray *, int);
    result = emalloc(sizeof(CArray));
    int * dimensions, i = 0, prior2_dimension = 0, dim_diff;
    int typenum;
    CArrayDescriptor * type;
    int inverted = 0;

    if(CArray_NDIM(m1) > CArray_NDIM(m2)) {
        prior1 = m1;
        prior2 = m2;
    } else if(CArray_NDIM(m1) == CArray_NDIM(m2)) {
        if (CArray_SIZE(m1) >= CArray_SIZE(m2)) {
            prior1 = m1;
            prior2 = m2;
        } else {
            prior1 = m2;
            prior2 = m1;
            inverted = 1;
        }
    }
    else {
        prior1 = m2;
        prior2 = m1;
        inverted = 1;
    }

    dim_diff = CArray_NDIM(prior1) - CArray_NDIM(prior2);
    dimensions = ecalloc(CArray_NDIM(prior1), sizeof(int));

    for(i = 0; i < CArray_NDIM(prior1); i++) {
        dimensions[i] = CArray_DIM(prior1, i);
    }

    type = CArray_DescrFromType(TYPE_DOUBLE_INT);

    switch(CArray_TYPE(prior1)) {
        case TYPE_DOUBLE_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_mod_double_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_mod_double_int;
                    break;
            }
            break;
        case TYPE_INTEGER_INT:
            switch(CArray_TYPE(prior2)) {
                case TYPE_DOUBLE_INT:
                    data_op = &_carray_mod_int_double;
                    break;
                case TYPE_INTEGER_INT:
                    data_op = &_carray_mod_int_int;
                    break;
            }
            break;
        default:
            break;
    }

    result = CArray_NewFromDescr_int(result, type, CArray_NDIM(prior1),  dimensions,
                                     NULL, NULL, 0, NULL, 1, 0);

    CArrayIterator * it1 = CArray_BroadcastToShape(prior1, dimensions, CArray_NDIM(prior1));
    CArrayIterator * it2 = CArray_BroadcastToShape(prior2, dimensions, CArray_NDIM(prior1));
    CArrayIterator * tmp_it;
    if (inverted) {
        tmp_it = it1;
        it1 = it2;
        it2 = tmp_it;
    }


    if (it1 == NULL || it2 == NULL) {
        return NULL;
    }

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