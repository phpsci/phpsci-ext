//
// Created by Henrique Borba on 19/11/2018.
//
#include "php.h"
#include "iterators.h"
#include "carray.h"
#include "flagsobject.h"

void 
CArray_ITER_RESET(CArrayIterator * iterator)
{
    do {
        iterator->index = 0;
        iterator->data_pointer = CArray_BYTES(iterator->array);
        memset(iterator->coordinates, 0, (iterator->ndims_m1+1)*sizeof(int));
    } while(0);
}

void 
_CArrayIterator_NEXT1(CArrayIterator * iterator)
{
    do {
        iterator->data_pointer += iterator->strides[0];
        iterator->coordinates[0]++;
    } while(0);
}

void 
_CArrayIterator_NEXT2(CArrayIterator * iterator)
{
    do {
        if(iterator->coordinates[1] < iterator->dims_m1[1]) {
            iterator->coordinates[1]++;
            iterator->data_pointer += iterator->strides[1];
        } else {
            iterator->coordinates[1] = 0;
            iterator->coordinates[0]++;
            iterator->data_pointer += iterator->strides[0] - iterator->backstrides[1];
        }
    } while(0);
}

void
CArrayIterator_GOTO(CArrayIterator * iterator, int * destination)
{
    int i;
    iterator->index = 0;
    iterator->data_pointer = CArray_BYTES(iterator->array);
    for (i = iterator->ndims_m1; i>=0; i--) {
        if (destination[i] < 0) {
            destination[i] += iterator->dims_m1[i]+1;
        }
        iterator->data_pointer += destination[i] * iterator->strides[i];
        iterator->coordinates[i] = destination[i];
        iterator->index += destination[i] * ( i == iterator->ndims_m1 ? 1 :iterator->dims_m1[i+1]+1) ;
    }
}

void
CArrayIterator_NEXT(CArrayIterator * iterator)
{
    do {
        iterator->index = iterator->index + 1;
        if(iterator->ndims_m1 == 0) {
            _CArrayIterator_NEXT1(iterator);
        }
        else if(iterator->contiguous) {
            iterator->data_pointer += CArray_DESCR(iterator->array)->elsize;
        }
        else if(iterator->ndims_m1 == 1) {
            _CArrayIterator_NEXT2(iterator);
        }
        else {
            int i;
            for(i = iterator->ndims_m1; i >= 0; i--) {
                if(iterator->coordinates[i] < iterator->dims_m1[i]) {
                    iterator->coordinates[i]++;
                    iterator->data_pointer += iterator->strides[i];
                    break;
                } else  {
                    iterator->coordinates[i] = 0;
                    iterator->data_pointer -= iterator->backstrides[i];
                }
            }
        }
    } while(0);
}

/**
 * Get DATA pointer from iterator
 *
 * @param iter
 * @param coordinates
 * @return
 */
static char*
get_ptr(CArrayIterator * iter, uintptr_t * coordinates)
{
    uintptr_t i;
    char *ret;

    ret = CArray_DATA(iter->array);

    for(i = 0; i <  CArray_NDIM(iter->array); ++i) {
        ret += coordinates[i] * iter->strides[i];
    }

    return ret;
}

/**
 * @param iterator
 */
void
CArrayIterator_Dump(CArrayIterator * iterator)
{
    int i;
    php_printf("CArrayIterator.index\t%d\n", iterator->index);
    php_printf("CArrayIterator.ndims_m1\t%d\n", iterator->ndims_m1);
    php_printf("CArrayIterator.factors\t\t[");
    for(i = 0; i < iterator->ndims_m1+1; i ++) {
        php_printf(" %d", iterator->factors[i]);
    }
    php_printf(" ]\n");
    php_printf("CArrayIterator.limits\t\t[");
    for(i = 0; i < iterator->ndims_m1+1; i ++) {
        php_printf(" [%d  %d]", iterator->limits[i][0], iterator->limits[i][1]);
    }
    php_printf(" ]\n");
    php_printf("CArrayIterator.bounds\t\t[");
    for(i = 0; i < iterator->ndims_m1+1; i ++) {
        php_printf(" [%d  %d]", iterator->bounds[i][0], iterator->bounds[i][1]);
    }
    php_printf(" ]\n");
    php_printf("CArrayIterator.backstrides\t\t[");
    for(i = 0; i < iterator->ndims_m1+1; i ++) {
        php_printf(" %d", iterator->backstrides[i]);
    }
    php_printf(" ]\n");
    php_printf("CArrayIterator.limits_sizes\t\t[");
    for(i = 0; i < iterator->ndims_m1+1; i ++) {
        php_printf(" %d", iterator->limits_sizes[i]);
    }
    php_printf(" ]\n");
}

/**
 * Base Iterator Initializer
 **/ 
void
iterator_base_init(CArrayIterator * iterator, CArray * array)
{
    int nd, i;
    nd = CArray_NDIM(array);

    CArray_UpdateFlags(array, CARRAY_ARRAY_C_CONTIGUOUS);
    if (CArray_ISCONTIGUOUS(array)) {
        iterator->contiguous = 1;
    } else {
        iterator->contiguous = 0;
    }
    
    iterator->array = array;
    iterator->size = array->descriptor->numElements;
    iterator->ndims_m1 = nd - 1;

    iterator->bounds = (int**)emalloc(nd * 2 * sizeof(int));
    for(i=0; i < nd; i++)
        iterator->bounds[i] = (int*)emalloc(2 * sizeof(int));

    iterator->limits = (int**)emalloc(nd * 2 * sizeof(int));

    for(i=0; i < nd; i++)
        iterator->limits[i] = (int*)emalloc(2 * sizeof(int));

    iterator->factors = (int*)emalloc(nd * sizeof(int));
    iterator->limits_sizes = (int*)emalloc(nd * sizeof(int));
    iterator->strides = (int*)emalloc(nd * sizeof(int));
    iterator->dims_m1 = (int*)emalloc(nd * sizeof(int));
    iterator->backstrides = (int*)emalloc(nd * sizeof(int));
    iterator->coordinates = (int*)emalloc(nd * sizeof(int));

    if (nd != 0) {
        iterator->factors[nd-1] = 1;
    }

    for (i = 0; i < nd; i++) {
        iterator->dims_m1[i] = CArray_DIMS(array)[i] - 1;
        iterator->strides[i] = CArray_STRIDES(array)[i];
        iterator->backstrides[i] = iterator->strides[i] * iterator->dims_m1[i];
        if (i > 0) {
            iterator->factors[nd-i-1] = iterator->factors[nd-i] * CArray_DIMS(array)[nd-i];
        }
        iterator->bounds[i][0] = 0;
        iterator->bounds[i][1] = CArray_DIMS(array)[i] - 1;
        iterator->limits[i][0] = 0;
        iterator->limits[i][1] = CArray_DIMS(array)[i] - 1;
        iterator->limits_sizes[i] = iterator->limits[i][1] - iterator->limits[i][0] + 1;
    }
    CArray_ITER_RESET(iterator);
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
    return iterator;
}