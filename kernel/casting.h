#ifndef PHPSCI_EXT_CASTING_H
#define PHPSCI_EXT_CASTING_H

#include "carray.h"

void DOUBLE_copyswap (void *dst, void *src, int swap, void * arr);
void INT_copyswap (void *dst, void *src, int swap, void * arr);

void DOUBLE_TO_INT(int *ip, double *op, int n, CArray *aip, CArray *aop);
void INT_TO_DOUBLE(int *ip, double *op, int n, CArray *aip, CArray *aop);

int DOUBLE_setitem (void * op, void * ov, struct CArray * ap);
int INT_setitem (int * op, void * ov, struct CArray * ap);

int DOUBLE_fill(void * op, void * ov, struct CArray * ap);
int INT_fill(int * op, void * ov, struct CArray * ap);
#endif