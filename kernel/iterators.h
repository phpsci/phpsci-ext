#ifndef PHPSCI_EXT_ITERATORS_H
#define PHPSCI_EXT_ITERATORS_H

#include "carray.h"

typedef struct CArrayIterator
{
    int    index;             // Current 1-d index
    int    size;
    int  * coordinates;     // Coordinate vectors of index
    int  * dims_m1;          // Size of array minus 1 for each dimension
    int    ndims_m1;
    int  * factors;         // Factor for ND-index to 1D-index
    int  * strides;         // Array Strides
    int  * backstrides;     // Backstrides
    char * data_pointer;    // Data pointer to element defined by index
    int    contiguous;      // 1 = Contiguous, 0 = Non-contiguous
    int  ** bounds;
    int  ** limits;
    int  *  limits_sizes;
    CArray * array;         // Pointer to represented CArray
} CArrayIterator;

#define IT_IDATA(it) ((int *)((it)->data_pointer))
#define IT_DDATA(it) ((double *)((it)->data_pointer))
#define CArrayIterator_DATA(it) ((void *)((it)->data_pointer))
#define CArrayIterator_NOTDONE(it) ((it)->index < (it)->size)

CArrayIterator * CArray_NewIter(CArray * array);
static char* get_ptr(CArrayIterator * iter, uintptr_t * coordinates);
void CArrayIterator_Dump(CArrayIterator * iterator);
void CArrayIterator_GOTO(CArrayIterator * iterator, int * destination);
void CArrayIterator_NEXT(CArrayIterator * iterator);
void CArrayIterator_RESET(CArrayIterator * iterator);
void CArrayIterator_FREE(CArrayIterator * it);

CArrayIterator * CArray_BroadcastToShape(CArray * target, int * dims, int nd);
CArrayIterator * CArray_IterAllButAxis(CArray *obj, int *inaxis);
#endif //PHPSCI_EXT_ITERATORS_H