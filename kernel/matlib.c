#include "carray.h"
#include "matlib.h"
#include "scalar.h"
#include "alloc.h"
#include "convert.h"

CArray *
CArray_Zeros(int * shape, int nd, char type, char * order, MemoryPointer * rtn_ptr)
{
    int is_fortran = 0;
    CArrayDescriptor * new_descr;
    CArrayScalar * sc = emalloc(sizeof(CArrayScalar));
    CArray * rtn;
    int allocated = 0;

    if (order == NULL) {
        order = emalloc(sizeof(char));
        *order = 'C';
        allocated = 1;
    }

    if (*order == 'F') {
        is_fortran = 1;
    }

    new_descr = CArray_DescrFromType(CHAR_TYPE_INT(type));
    rtn = CArray_Empty(nd, shape, new_descr, is_fortran, rtn_ptr);

    sc->obval = emalloc(new_descr->elsize);
    sc->type  = CHAR_TYPE_INT(type);

    if(type == TYPE_DOUBLE){
        *((double *)sc->obval) = (double)0.00;
    }
    if(type == TYPE_INTEGER){
        *((int *)sc->obval) = (int)0;
    }
    if(type == TYPE_FLOAT){
        *((float *)sc->obval) = (float)0;
    }
    
    
    CArray_FillWithScalar(rtn, sc);

    efree(sc->obval);
    efree(sc);

    if(allocated) {
        efree(order);
    }
    return rtn;
}


CArray *
CArray_Ones(int * shape, int nd, char * type, char * order, MemoryPointer * rtn_ptr)
{
    int is_fortran = 0;
    CArrayDescriptor * new_descr;
    CArrayScalar * sc = emalloc(sizeof(CArrayScalar));
    CArray * rtn;

    if (order == NULL) {
        order = emalloc(sizeof(char));
        *order = 'C';
    }

    if (*order == 'F') {
        is_fortran = 1;
    }

    new_descr = CArray_DescrFromType(CHAR_TYPE_INT(*type));
    rtn = CArray_Empty(nd, shape, new_descr, is_fortran, rtn_ptr);

    sc->obval = emalloc(new_descr->elsize);
    sc->type  = CHAR_TYPE_INT(*type);

    if(*type == TYPE_DOUBLE){
        *((double *)sc->obval) = (double)1.00;
    }
    if(*type == TYPE_INTEGER){
        *((int *)sc->obval) = (int)1;
    }
    if(*type == TYPE_FLOAT){
        *((float *)sc->obval) = (float)1;
    }
    
    CArray_FillWithScalar(rtn, sc);

    efree(sc->obval);
    efree(sc);
    return rtn;
}