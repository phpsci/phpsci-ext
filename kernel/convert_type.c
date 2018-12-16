#include "convert_type.h"
#include "alloc.h"
#include "carray.h"

/**
 * @param op
 * @param minimum_type
 * @return
 */
int
CArray_ObjectType(CArray * op, int minimum_type)
{
    CArrayDescriptor *dtype = NULL;
    int ret;

    if (minimum_type >= 0) {
        dtype = CArray_DescrFromType(minimum_type);

        if (dtype == NULL) {
            return TYPE_NOTYPE_INT;
        }
    }

    if (dtype == NULL) {
        ret = TYPE_DEFAULT_INT;
    }
    else {
        ret = dtype->type_num;
    }

    CArrayDescriptor_DECREF(dtype);

    return ret;
}