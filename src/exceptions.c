#include "exceptions.h"
#include "php.h"
#include "Zend/zend_exceptions.h"
#include <Python.h>

static zend_class_entry * phpsci_ce_PythonException;

static const zend_function_entry phpsci_ce_PythonException_methods[] = {
        PHP_FE_END
};


/**
 * Initialize Exception Classes
 */
void
init_exception_objects()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "PythonException", phpsci_ce_PythonException_methods);
    phpsci_ce_PythonException = zend_register_internal_class_ex(&ce, zend_ce_exception);
}

void
throw_python_exception()
{
    PyObject *ptype, *pvalue, *ptraceback;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if(pvalue) {
        PyObject *pstr = PyObject_Str(pvalue);
        if(pstr) {
            const char* err_msg = PyUnicode_AsUTF8(pstr);
            if(pstr){
                zend_throw_exception_ex(phpsci_ce_PythonException, 0, "%s",
                                        err_msg );
            }
        }
        PyErr_Restore(ptype, pvalue, ptraceback);
        PyErr_Clear();
    }
}
