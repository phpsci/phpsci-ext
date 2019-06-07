#include "trigonometric.h"
#include "carray.h"
#include "buffer.h"

void
CArray_ElementWise_CFunc_Double(CArray * target, CArray * result, CArray_CFunc_ElementWise * func)
{
    int i;
    for(i = 0; i < CArray_DESCR(target)->numElements; i++) {
        DDATA(result)[i] = func(DDATA(target)[i]);
    }
}

void
CArray_ElementWise_CFunc_Int(CArray * target, CArray * result, CArray_CFunc_ElementWise * func)
{
    int i;
    for(i = 0; i < CArray_DESCR(target)->numElements; i++) {
        DDATA(result)[i] = func(IDATA(target)[i]);
    }
}

void
CArray_ElementWise_CFunc(CArray * target, CArray * result, CArray_CFunc_ElementWise * func)
{
    if(CArray_TYPE(target) == TYPE_INTEGER_INT) {
        CArray_ElementWise_CFunc_Int(target, result, func);
        return;
    }
    if(CArray_TYPE(target) == TYPE_DOUBLE_INT) {
        CArray_ElementWise_CFunc_Double(target, result, func);
        return;
    }
}

CArray *
CArray_Sin(CArray * target, MemoryPointer * out)
{
    CArray * result;
    CArrayDescriptor * descr;
    int * new_strides;
    result = emalloc(sizeof(CArray));

    descr = CArray_DescrFromType(TYPE_DOUBLE_INT);
    new_strides = CArray_Generate_Strides(CArray_DIMS(target), CArray_NDIM(target), descr->type);

    result = CArray_NewFromDescr_int(result, descr, CArray_NDIM(target),
            CArray_DIMS(target), new_strides, NULL,
            CArray_FLAGS(target), NULL, 1, 0);
    result->flags = ~CARRAY_ARRAY_F_CONTIGUOUS;
    CArray_ElementWise_CFunc(target, result, &sin);

    if(out != NULL) {
        add_to_buffer(out, result, sizeof(CArray *));
    }
    efree(new_strides);
}
