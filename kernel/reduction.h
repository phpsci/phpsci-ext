#ifndef PHPSCI_EXT_REDUCTION_H
#define PHPSCI_EXT_REDUCTION_H

#include "carray.h"
#include "iterators.h"


typedef int (CArray_ReduceLoopFunc)(CArrayIterator *iter,
                                            char **dataptr,
                                            int *strideptr,
                                            int *countptr,
                                            CArrayIterator_IterNextFunc *iternext,
                                            int needs_api,
                                            int skip_first_count,
                                            void *data);

#endif