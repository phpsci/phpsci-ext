#include "carray.h"
#include "alloc.h"
#include "buffer.h"
#include "range.h"
#include "convert.h"
#include "number.h"

/**
 * @param start
 * @param stop
 * @param step
 * @param type_num
 * @param ptr
 * @todo Fix leak
 * @return
 */
CArray *
CArray_Arange(double start, double stop, double step, int type_num, MemoryPointer * ptr)
{
    int length;
    CArray * range;
    CArray_ArrFuncs *funcs;
    int    start_plus_step_i, start_i;
    double start_plus_step_d, start_d;
    int ret;

    range = emalloc(sizeof(CArray));

    if (_safe_ceil_to_int((stop - start) / step, &length)) {
        throw_overflow_exception("arange: overflow while computing length");
    }

    if (length <= 0) {
        length = 0;
        if(ptr != NULL) {
            add_to_buffer(ptr, range, sizeof(CArray));
        }
        return CArray_New(range, 1, &length, type_num,
                          NULL, NULL, 0, 0, NULL);
    }

    range = CArray_New(range, 1, &length, type_num,
                       NULL, NULL, 0, 0, NULL);

    if (range == NULL) {
        return NULL;
    }

    funcs = CArray_DESCR(range)->f;

    if(type_num == TYPE_DOUBLE_INT) {
        start_d = (double)(start);
        ret = funcs->setitem(((double *) &start_d), CArray_BYTES(range), range);
    }
    if(type_num == TYPE_INTEGER_INT) {
        start_i = (int)(start);
        ret = funcs->setitem(((int *) &start_i), CArray_BYTES(range), range);
    }

    if (ret < 0) {
        goto fail;
    }

    if(ptr != NULL) {
        add_to_buffer(ptr, range, sizeof(CArray));
    }

    if (length == 1) {
        return range;
    }

    if(type_num == TYPE_DOUBLE_INT) {
        start_plus_step_d = (double)(start + step);
        ret = funcs->setitem(&start_plus_step_d, (CArray_BYTES(range) + CArray_ITEMSIZE(range)), range);
    }
    if(type_num == TYPE_INTEGER_INT) {
        start_plus_step_i = (int)(start + step);
        ret = funcs->setitem(&start_plus_step_i, (CArray_BYTES(range) + CArray_ITEMSIZE(range)), range);
    }

    if (ret < 0) {
        goto fail;
    }

    if (length == 2) {
        return range;
    }

    if (funcs->fill == NULL) {
        throw_valueerror_exception("no fill-function for data-type.");
        return NULL;
    }

    funcs->fill(CArray_BYTES(range), length, range);
    return range;
fail:
    return NULL;
}


CArray *
CArray_Linspace(double start, double stop, int num, int endpoint, int retstep, int axis, int type, MemoryPointer * out)
{
    CArray * y, * tempc = emalloc(sizeof(CArray)), * freec, * rtn;
    double div, delta;
    CArrayDescriptor * dtype;
    double step;
    if(num < 0)  {
        throw_valueerror_exception("Number of samples must be non-negative.");
        return NULL;
    }

    if(endpoint) {
        div = (num - 1);
    } else {
        div = num;
    }

    dtype = CArray_DescrFromType(TYPE_DOUBLE_INT);
    tempc = CArray_NewFromDescr(tempc, dtype, 0, NULL, NULL, NULL,
            0, NULL);

    DDATA(tempc)[0] = div;

    delta = stop - start;
    y = CArray_Arange(0.0, num, 1.0, type, NULL);
    freec = y;

    if(num > 1) {
        step = delta / div;
        if (step == 0) {
            y = CArray_Divide(y, tempc, NULL);
            DDATA(tempc)[0] = delta;
            y = CArray_Multiply(y, tempc, NULL);
        } else {
            DDATA(tempc)[0] = step;
            y = CArray_Multiply(y, tempc, NULL);
            CArray_Free(freec);
        }
    } else {
        DDATA(tempc)[0] = delta;
        y = CArray_Multiply(y, tempc, NULL);
    }

    freec = y;
    DDATA(tempc)[0] = start;

    y = CArray_Add(y, tempc, NULL);
    CArray_Free(freec);

    if (endpoint && num > 1) {
        if (CArray_TYPE(y) == TYPE_INTEGER_INT) {
            IDATA(y)[CArray_DESCR(y)->numElements-1] = (int)stop;
        }
        if (CArray_TYPE(y) == TYPE_DOUBLE_INT) {
            DDATA(y)[CArray_DESCR(y)->numElements-1] = (double)stop;
        }
    }

    CArrayDescriptor_FREE(dtype);
    CArray_Free(tempc);

    if (type != TYPE_DOUBLE_INT) {
        CArrayDescriptor *descr = CArray_DescrFromType(type);
        rtn = CArray_NewLikeArray(y, CARRAY_CORDER, descr, 0);
        if(CArray_CastTo(rtn, y) < 0) {
            return NULL;
        }
        CArray_Free(y);
    } else {
        rtn = y;
    }

    if (out != NULL) {
        add_to_buffer(out, rtn, sizeof(CArray));
    }

    return rtn;
}

