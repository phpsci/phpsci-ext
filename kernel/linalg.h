#ifndef PHPSCI_EXT_LINALG_H
#define PHPSCI_EXT_ITERATORS_H

#define INIT_OUTER_LOOP_1 \
    npy_intp dN = *dimensions++;\
    npy_intp N_;\
    npy_intp s0 = *steps++;

#define INIT_OUTER_LOOP_2 \
    INIT_OUTER_LOOP_1\
    npy_intp s1 = *steps++;

#define INIT_OUTER_LOOP_3 \
    INIT_OUTER_LOOP_2\
    npy_intp s2 = *steps++;

#define INIT_OUTER_LOOP_4 \
    INIT_OUTER_LOOP_3\
    npy_intp s3 = *steps++;

#define INIT_OUTER_LOOP_5 \
    INIT_OUTER_LOOP_4\
    npy_intp s4 = *steps++;

#define INIT_OUTER_LOOP_6  \
    INIT_OUTER_LOOP_5\
    npy_intp s5 = *steps++;

#define INIT_OUTER_LOOP_7  \
    INIT_OUTER_LOOP_6\
    npy_intp s6 = *steps++;

#define BEGIN_OUTER_LOOP_2 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1) {

#define BEGIN_OUTER_LOOP_3 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2) {

#define BEGIN_OUTER_LOOP_4 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3) {

#define BEGIN_OUTER_LOOP_5 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4) {

#define BEGIN_OUTER_LOOP_6 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5) {

#define BEGIN_OUTER_LOOP_7 \
    for (N_ = 0;\
         N_ < dN;\
         N_++, args[0] += s0,\
             args[1] += s1,\
             args[2] += s2,\
             args[3] += s3,\
             args[4] += s4,\
             args[5] += s5,\
             args[6] += s6) {

#define END_OUTER_LOOP  }

typedef void (CArray_DotFunc)(char *, int, char *, int, char *, int);

void FLOAT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);
void INT_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);
void DOUBLE_dot(char *ip1, int is1, char *ip2, int is2, char *op, int n);

CArray * CArray_Matmul(CArray * ap1, CArray * ap2, CArray * out, MemoryPointer * ptr);

#endif //PHPSCI_EXT_ITERATORS_H