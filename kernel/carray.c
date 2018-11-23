//
// Created by Henrique Borba on 19/11/2018.
//
#include "carray.h"
#include "alloc.h"
#include "iterators.h"
#include "buffer.h"
#include "php.h"
#include "php_ini.h"
#include "zend_smart_str.h"
#include "zend_bitset.h"
#include "ext/spl/spl_array.h"
#include "zend_globals.h"
#include "zend_interfaces.h"
#include "php_ini.h"
#include "php_variables.h"
#include "php_globals.h"
#include "php_content_types.h"
#include "zend_multibyte.h"
#include "zend_smart_str.h"

/**
 * @param CHAR_TYPE
 */
int
CHAR_TYPE_INT(char CHAR_TYPE)
{
    if(CHAR_TYPE == TYPE_DOUBLE) {
        return 1;
    }
    if(CHAR_TYPE == TYPE_INTEGER) {
        return 2;
    }
}

/**
 * Print current CArray
 **/ 
void
CArray_ToString(CArray * carray)
{

}


/**
 * Create CArray from Double ZVAL
 * @return
 */
MemoryPointer
CArray_FromZval_Double(zval * php_obj, char * type)
{

}

/**
 * Create CArray from Long ZVAL
 * @return
 */
MemoryPointer
CArray_FromZval_Long(zval * php_obj, char * type)
{

}

/**
 * @param dims
 * @param ndims
 * @param type
 */
int *
CArray_Generate_Strides(int * dims, int ndims, char type)
{
    int i;
    int * strides;
    int * target_stride = (int*)emalloc(ndims * sizeof(int));

    for(i = 0; i < ndims; i++) {
        target_stride[i] = 0;
    }

    if(type == TYPE_INTEGER)
        target_stride[ndims-1] = sizeof(int);
    if(type == TYPE_DOUBLE)
        target_stride[ndims-1] = sizeof(double);

    for(i = ndims-2; i >= 0; i--) {
        target_stride[i] = dims[i+1] * target_stride[i+1];
    }
    return target_stride;
}

/**
 * Walk trought all PHP values
 * @return
 */
zval * php_array_values_walk(zval * php_array)
{
    zval * zv;
}

/**
 *
 * @param target_zval
 * @param ndims
 */
void
Hashtable_ndim(zval * target_zval, int * ndims)
{
    zval * element;
    int current_dim = *ndims;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
        ZVAL_DEREF(element);
        if (Z_TYPE_P(element) == IS_ARRAY) {
            *ndims = *ndims + 1;
            Hashtable_ndim(element, ndims);
            break;
        }
    } ZEND_HASH_FOREACH_END();
}

/**
 *
 * @param target_zval
 * @param dims
 * @param ndims
 * @param current_dim
 */
void
Hashtable_dimensions(zval * target_zval, int * dims, int ndims, int current_dim)
{
    zval * element;
    if (Z_TYPE_P(target_zval) == IS_ARRAY) {
        dims[current_dim] = (int)zend_hash_num_elements(Z_ARRVAL_P(target_zval));
    }

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
        ZVAL_DEREF(element);
        if (Z_TYPE_P(element) == IS_ARRAY) {
            Hashtable_dimensions(element, dims, ndims, (current_dim + 1));
            break;
        }
    } ZEND_HASH_FOREACH_END();
}

/**
 *
 * @param target_zval
 * @param dims
 * @param ndims
 * @param current_dim
 */
void
Hashtable_type(zval * target_zval, char * type)
{
    zval * element;

    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
        ZVAL_DEREF(element);
        if (Z_TYPE_P(element) == IS_ARRAY) {
            Hashtable_type(element, type);
            break;
        }
        if (Z_TYPE_P(element) == IS_LONG) {
            *type = 'i';
            break;
        }
        if (Z_TYPE_P(element) == IS_DOUBLE) {
            *type = 'd';
            break;
        }
        if (Z_TYPE_P(element) == IS_STRING) {
            *type = 'c';
            break;
        }
    } ZEND_HASH_FOREACH_END();
}

/**
 * Create CArray from HashTable ZVAL
 * @param php_array
 * @param type
 * @return
 */
void
CArray_FromZval_Hashtable(zval * php_array, char * type, MemoryPointer * ptr)
{
    CArray new_carray;
    int * dims, ndims = 1;
    int last_index = 0;
    Hashtable_ndim(php_array, &ndims);
    dims = (int*)emalloc(ndims * sizeof(int));
    Hashtable_dimensions(php_array, dims, ndims, 0);
    // If `a` (auto), find best element type
    if(!strcmp("a", type)) {
        Hashtable_type(php_array, type);
    }

    CArray_INIT(ptr, &new_carray, dims, ndims, *type);
    CArray_Hashtable_Data_Copy(&new_carray, php_array, &last_index);
}

/**
 * Dump CArray
 */
void
CArray_Dump(CArray * ca)
{
    int i;
    php_printf("CArray.dims\t\t\t[");
    for(i = 0; i < ca->ndim; i ++) {
        php_printf(" %d", ca->dimensions[i]);
    }
    php_printf(" ]\n");
    php_printf("CArray.strides\t\t\t[");
    for(i = 0; i < ca->ndim; i ++) {
        php_printf(" %d", ca->strides[i]);
    }
    php_printf(" ]\n");
    php_printf("CArray.ndim\t\t\t%d\n", ca->ndim);
    php_printf("CArray.descriptor.elsize\t%d\n", ca->descriptor->elsize);
    php_printf("CArray.descriptor.numElements\t%d\n", ca->descriptor->numElements);
    php_printf("CArray.descriptor.type\t\t%c\n", ca->descriptor->type);
    php_printf("CArray.descriptor.type_num\t%d\n", ca->descriptor->type_num);
    php_printf("CArray.descriptor.elsize\t%d\n", ca->descriptor->elsize);
}

/**
 * Multiply vector list by scalar
 *
 * @param list
 * @param scalar
 * @return
 */
int
CArray_MultiplyList(const int * list, unsigned int size)
{
    int i;
    int total = 0;
    for(i = size; i >= 0; i--) {
        if(i != size)
            total += list[i] * list[i+1];
    }
    return total;
}

/**
 * @param target_carray
 */
void
CArray_Hashtable_Data_Copy(CArray * target_carray, zval * target_zval, int * first_index)
{
    zval * element;
    int * data_int;
    double * data_double;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(target_zval), element) {
        ZVAL_DEREF(element);
        if (Z_TYPE_P(element) == IS_ARRAY) {
            CArray_Hashtable_Data_Copy(target_carray, element, first_index);
        }
        if (Z_TYPE_P(element) == IS_LONG) {
            convert_to_long(element);
            data_int = (int*)CArray_DATA(target_carray);
            data_int[*first_index] = (int)zval_get_long(element);
            *first_index = *first_index + 1;
        }
        if (Z_TYPE_P(element) == IS_DOUBLE) {
            convert_to_double(element);
            data_double = (double*)CArray_DATA(target_carray);
            data_double[*first_index] = (double)zval_get_double(element);
            *first_index = *first_index + 1;
        }
        if (Z_TYPE_P(element) == IS_STRING) {

        }
    } ZEND_HASH_FOREACH_END();
}

/**
 * @param ptr
 * @param num_elements
 * @param dims
 * @param ndim
 */
void
CArray_INIT(MemoryPointer * ptr, CArray * output_ca, int * dims, int ndim, char type)
{
    CArrayDescriptor * output_ca_dscr;
    int * target_stride;
    int i, num_elements = 0;
    for(i = 0; i < ndim; i++) {
        if(i == 0) {
            num_elements = dims[i];
            continue;
        }
        num_elements = dims[i] * num_elements;
    }
    target_stride = CArray_Generate_Strides(dims, ndim, type);
    output_ca_dscr = (CArrayDescriptor*)emalloc(sizeof(CArrayDescriptor));
    // Build CArray Data Descriptor
    output_ca_dscr->type = type;
    output_ca_dscr->elsize = target_stride[ndim-1];
    output_ca_dscr->type_num = CHAR_TYPE_INT(type);
    output_ca_dscr->numElements = num_elements;

    // Build CArray
    
    output_ca->descriptor = output_ca_dscr;
    output_ca->dimensions = dims;
    output_ca->ndim = ndim;
    output_ca->strides = target_stride;
    CArray_Data_alloc(output_ca);
    add_to_buffer(ptr, *output_ca, sizeof(output_ca));
}

/**
 * Create CArray from ZVAL
 * @return MemoryPointer
 */
void
CArray_FromZval(zval * php_obj, char * type, MemoryPointer * ptr)
{
    if(Z_TYPE_P(php_obj) == IS_LONG) {
        php_printf("LONG");
    }
    if(Z_TYPE_P(php_obj) == IS_ARRAY) {
        CArray_FromZval_Hashtable(php_obj, type, ptr);
    }
    if(Z_TYPE_P(php_obj) == IS_DOUBLE) {
        php_printf("DOUBLE");
    }
}

/**
 * Convert MemoryPointer to CArray
 * @param ptr
 */
CArray *
CArray_FromMemoryPointer(MemoryPointer * ptr)
{
    return &(PHPSCI_MAIN_MEM_STACK.buffer[ptr->uuid]);
}