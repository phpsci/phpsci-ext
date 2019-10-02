#ifndef CARRAY_CARRAY_H
#define CARRAY_CARRAY_H

#include "php.h"
#include <Python.h>

typedef struct MemoryPointer {
    int uuid;
    int free;
} MemoryPointer;

/**
 *
 */
typedef struct tagCArray {
    PyObject_HEAD
} CArray;


CArray * CArray_NewFromArrayPHP(zval *arr, int type);
CArray * CArray_Print(CArray *a);
void CArray_Dump(CArray *a);
CArray * CArray_FromMemoryPointer(MemoryPointer *ptr);

CArray * CArray_Ones(CArray *shape, int typenum);
CArray * CArray_Zeros(CArray *shape, int typenum);

CArray * CArray_Arange(double start, double stop, double step, int typenum);
#endif //CARRAY_CARRAY_H
