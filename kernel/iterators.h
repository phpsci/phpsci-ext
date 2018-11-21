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

CArrayIterator * CArray_NewIter(CArray * array);
static char* get_ptr(CArrayIterator * iter, uintptr_t * coordinates);
void CArrayIterator_Dump(CArrayIterator * iterator);
#endif //PHPSCI_EXT_ITERATORS_H