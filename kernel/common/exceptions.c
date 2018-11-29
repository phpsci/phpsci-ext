#include "php.h"
#include "exceptions.h"
#include "Zend/zend_exceptions.h"

static zend_class_entry * phpsci_ce_CArrayAxisException;

static const zend_function_entry phpsci_ce_CArrayAxisException_methods[] = {
        PHP_FE_END
};

/**
 * Initialize Exception Classes
 */
void
init_exception_objects()
{
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "CArrayAxisException", phpsci_ce_CArrayAxisException_methods);
    phpsci_ce_CArrayAxisException = zend_register_internal_class_ex(&ce, zend_ce_exception);
}

/**
 * Throw CArrayAxisException
 */
void
throw_axis_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayAxisException, AXIS_EXCEPTION, "%s", msg);
}
