//
// Created by Henrique Borba on 19/11/2018.
//

#ifndef PHPSCI_EXT_CARRAY_H
#define PHPSCI_EXT_CARRAY_H

#include "php.h"


#define TYPE_INTEGER     'i'
#define TYPE_DOUBLE      'd'
#define TYPE_INTEGER_INT  2
#define TYPE_DOUBLE_INT   1

/**
 * CArray Descriptor
 */
typedef struct CArrayDescriptor {
    char type;          // b = boolean, d = double, i = signer integer, u = unsigned integer, f = floating point, c = char
    int flags;          // Data related flags
    int type_num;       // 0 = boolean, 1 = double, 2 = signed integer, 3 = unsigned integer, 4 = floating point, 5 = char
    int elsize;         // Datatype size
} CArrayDescriptor;

/**
 * CArray
 */
typedef struct CArray CArray;
struct CArray {
    int * strides;      // Strides vector
    int * dimensions;   // Dimensions size vector (Shape)
    int ndim;           // Number of Dimensions
    char * data;        // Data Buffer
    CArray * base;      // Used when sharing memory from other CArray (slices, etc)
    int flags;          // Describes CArray memory approach (Memory related flags)
    CArrayDescriptor descriptor;    // CArray data descriptor
};

/**
 * Memory Pointer
 */
typedef struct MemoryPointer {
    int uuid;
} MemoryPointer;


int CHAR_TYPE_INT(char CHAR_TYPE);

void CArray_INIT(MemoryPointer * ptr, int * dims, int ndim, char type);
void CArray_FromZval(zval * php_obj, char * type);
void CArray_Dump(CArray ca);
#endif //PHPSCI_EXT_CARRAY_H
