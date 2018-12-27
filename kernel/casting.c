#include "carray.h"

/**
 * COPYSWAP DOUBLE
 **/ 
void
DOUBLE_copyswap (void *dst, void *src, int swap, void * arr)
{
    if (src != NULL) {
        /* copy first if needed */
        memcpy(dst, src, sizeof(double));
    }
    /* ignore swap */
}

/**
 * COPYSWAP INT
 **/ 
void
INT_copyswap (void *dst, void *src, int swap, void * arr)
{
    if (src != NULL) {
        /* copy first if needed */
        memcpy(dst, src, sizeof(int));
    }
    /* ignore swap */
}

/**
 * CAST DOUBLE TO INT
 **/ 
void
DOUBLE_TO_INT(double *ip, int *op, int n,
    CArray *aip, CArray *aop) {
    while (n--) {
        *(op++) = (int)*(ip++);
    }                
}

/**
 * CAST INT TO DOUBLE
 */ 
void
INT_TO_DOUBLE(int *ip, double *op, int n,
    CArray *aip, CArray *aop) {
    while (n--) {
        *(op++) = (double)*(ip++);
    }                
}