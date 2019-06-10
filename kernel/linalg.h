#ifndef PHPSCI_EXT_LINALG_H
#define PHPSCI_EXT_LINALG_H

typedef void (CArray_DotFunc)(char *, int, char *, int, char *, int);

void FLOAT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);
void INT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);
void DOUBLE_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);

CArray * CArray_Matmul(CArray * ap1, CArray * ap2, CArray * out, MemoryPointer * ptr);
CArray * CArray_Inv(CArray * a, MemoryPointer * out);
CArray * CArray_Norm(CArray * a, int norm, MemoryPointer * out);
#endif