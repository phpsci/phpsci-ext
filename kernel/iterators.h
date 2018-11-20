#ifndef PHPSCI_EXT_ITERATORS_H
#define PHPSCI_EXT_ITERATORS_H

#include "carray.h"

typedef struct CArrayIterator
{
    int  index;             // Current 1-d index
    int  * coordinates;     // Coordinate vectors of index
    int  * dim_m1;          // Size of array minus 1 for each dimension
    int  * factors;         // Factor for ND-index to 1D-index
    int  * strides;         // Array Strides
    int  * backstrides;     // Backstrides
    char * data_pointer;    // Data pointer to element defined by index
    int    contiguous;      // 1 = Contiguous, 0 = Non-contiguous
    CArray * array;         // Pointer to represented CArray
} CArrayIterator;

CArrayIterator * CArray_NewIter(CArray * array);

#endif //PHPSCI_EXT_ITERATORS_H