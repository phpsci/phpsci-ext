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

/*
 * Means c-style contiguous (last index varies the fastest). The data
 * elements right after each other.
 *
 * This flag may be requested in constructor functions.
 */
#define CARRAY_ARRAY_C_CONTIGUOUS    0x0001

/*
 * Set if array is a contiguous Fortran array: the first index varies
 * the fastest in memory (strides array is reverse of C-contiguous
 * array)
 *
 * This flag may be requested in constructor functions.
 */
#define CARRAY_ARRAY_F_CONTIGUOUS    0x0002

/*
 * Note: all 0-d arrays are C_CONTIGUOUS and F_CONTIGUOUS. If a
 * 1-d array is C_CONTIGUOUS it is also F_CONTIGUOUS. Arrays with
 * more then one dimension can be C_CONTIGUOUS and F_CONTIGUOUS
 * at the same time if they have either zero or one element.
 * If NPY_RELAXED_STRIDES_CHECKING is set, a higher dimensional
 * array is always C_CONTIGUOUS and F_CONTIGUOUS if it has zero elements
 * and the array is contiguous if ndarray.squeeze() is contiguous.
 * I.e. dimensions for which `ndarray.shape[dimension] == 1` are
 * ignored.
 */
#define CARRAY_ARRAY_OWNDATA         0x0004

/*
 * Array data is aligned on the appropriate memory address for the type
 * stored according to how the compiler would align things (e.g., an
 * array of integers (4 bytes each) starts on a memory address that's
 * a multiple of 4)
 *
 * This flag may be requested in constructor functions.
 */
#define CARRAY_ARRAY_ALIGNED         0x0100

/**
 * CArray Descriptor
 */
typedef struct CArrayDescriptor {
    char type;          // b = boolean, d = double, i = signer integer, u = unsigned integer, f = floating point, c = char
    int flags;          // Data related flags
    int type_num;       // 0 = boolean, 1 = double, 2 = signed integer, 3 = unsigned integer, 4 = floating point, 5 = char
    int elsize;         // Datatype size
    int numElements;    // Number of elements
    int alignment;      // Alignment Information
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
    CArrayDescriptor * descriptor;    // CArray data descriptor
};

/**
 * Memory Pointer
 */
typedef struct MemoryPointer {
    int uuid;
} MemoryPointer;

/**
 * Flags Object
 */ 
typedef struct CArrayFlags
{
    CArray * array;
    int      flags;
} CArrayFlags;


/**
 * CArray Data Macros
 **/ 
#define CHDATA(p) ((char *) CArray_DATA((CArray *)p))
#define SHDATA(p) ((short int *) CArray_DATA((CArray *)p))
#define DDATA(p) ((double *) CArray_DATA((CArray *)p))
#define FDATA(p) ((float *) CArray_DATA((CArray *)p))
#define CDATA(p) ((f2c_complex *) CArray_DATA((CArray *)p))
#define ZDATA(p) ((f2c_doublecomplex *) CArray_DATA((CArray *)p))
#define IDATA(p) ((int *) CArray_DATA((CArray *)p))

/**
 * CArrays Func Macros
 **/
#define CArray_BYTES(a) (a->data)
#define CArray_DATA(a) ((void *)((a)->data))
#define CArray_ITEMSIZE(a) ((int)((a)->descriptor->elsize))
#define CArray_DIMS(a) ((int *)((a)->dimensions))
#define CArray_STRIDES(a) ((int *)((a)->strides))
#define CArray_DESCR(a) ((a)->descriptor)

static inline int
CArray_FLAGS(const CArray *arr)
{
    return arr->flags;
}

static inline int
CArray_CHKFLAGS(const CArray *arr, int flags) {
    return (CArray_FLAGS(arr) & flags) == flags;
}

static inline int
CArray_NDIM(const CArray *arr) {
    return arr->ndim;
}
int CArray_MultiplyList(const int * list, unsigned int size);

#define CArray_SIZE(m) CArray_MultiplyList(CArray_DIMS(m), CArray_NDIM(m))
#define CArray_ISCONTIGUOUS(m) CArray_CHKFLAGS(m, CARRAY_ARRAY_C_CONTIGUOUS)

int CHAR_TYPE_INT(char CHAR_TYPE);
void CArray_INIT(MemoryPointer * ptr, CArray * output_ca, int * dims, int ndim, char type);
void CArray_Hashtable_Data_Copy(CArray * target_carray, zval * target_zval, int * first_index);
void CArray_FromZval(zval * php_obj, char * type, MemoryPointer * ptr);
void CArray_Dump(CArray * ca);
#endif //PHPSCI_EXT_CARRAY_H
