//
// Created by Henrique Borba on 19/11/2018.
//

#include "iterators.h"
#include "carray.h"

/**
 * Base Iterator Initializer
 **/ 
void
iterator_base_init(CArrayIterator * iterator, CArray * array)
{
    int nd, i;

    nd = CArray_NDIM(array);
    
}

/**
 * Return array iterator from CArray
 **/ 
CArrayIterator * 
CArray_NewIter(CArray * array)
{
    CArrayIterator * iterator;
    iterator = (CArrayIterator *)emalloc(sizeof(CArrayIterator));
    iterator_base_init(iterator, array);
}