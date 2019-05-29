#include "carray.h"

/**
 * FILL INT
 */
int
INT_fill(void * buffer, int length, struct CArray * ap)
{
    int i;
    int start = ((int*)buffer)[0];
    int delta = ((int*)buffer)[1];

    delta -= start;
    for (i = 2; i < length; ++i) {
        ((int*)buffer)[i] = start + i*delta;
    }
}

/**
 * FILL DOUBLE
 */
int
DOUBLE_fill(void * buffer, int length, struct CArray * ap)
{
    int i;
    double start = ((double*)buffer)[0];
    double delta = ((double*)buffer)[1];

    delta -= start;
    for (i = 2; i < length; ++i) {
        ((double*)buffer)[i] = start + i*delta;
    }
}

/**
 * SETITEM INT
 */
int
INT_setitem (void * op, void * ov, struct CArray * ap)
{
    int temp;  /* ensures alignment */

    temp = *((int *)op);
    if (ap == NULL || CArray_ISBEHAVED(ap))
        *((int *)ov)=temp;
    else {
        CArray_DESCR(ap)->f->copyswap(ov, &temp, !CArray_ISNOTSWAPPED(ap), ap);
    }
    return 0;
}

/**
 * GETITEM INT
 */

/**
 * SETITEM DOUBLE
 */
int
DOUBLE_setitem (double * op, void * ov, struct CArray * ap)
{
    double temp;  /* ensures alignment */

    temp = (double)*((double*)op);

    if (ap == NULL || CArray_ISBEHAVED(ap))
        *((double *)ov)=temp;
    else {
        CArray_DESCR(ap)->f->copyswap(ov, &temp, !CArray_ISNOTSWAPPED(ap), ap);
    }
    return 0;
}

/**
 * GETITEM DOUBLE
 */

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