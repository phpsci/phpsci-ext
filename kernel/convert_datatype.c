#include "carray.h"
#include "convert_datatype.h"
#include "common/exceptions.h"

int 
can_cast_scalar_to(CArrayDescriptor *scal_type, char *scal_data, CArrayDescriptor *to, CARRAY_CASTING casting)
{
    int swap;
    int is_small_unsigned = 0, type_num;
    int ret;
    CArrayDescriptor *dtype;

    /* An aligned memory buffer large enough to hold any type */
    long long value[4];

    /*
     * If the two dtypes are actually references to the same object
     * or if casting type is forced unsafe then always OK.
     */
    if (scal_type == to || casting == CARRAY_UNSAFE_CASTING ) {
        return 1;
    }

    throw_valueerror_exception("CASTING FATAL ERROR");
}
                       
                       