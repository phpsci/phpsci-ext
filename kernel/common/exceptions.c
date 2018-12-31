#include "php.h"
#include "exceptions.h"
#include "Zend/zend_exceptions.h"

static zend_class_entry * phpsci_ce_CArrayAxisException;
static zend_class_entry * phpsci_ce_CArrayValueErrorException;
static zend_class_entry * phpsci_ce_CArrayTypeErrorException;
static zend_class_entry * phpsci_ce_CArrayOverflowException;

static const zend_function_entry phpsci_ce_CArrayAxisException_methods[] = {
        PHP_FE_END
};
static const zend_function_entry phpsci_ce_CArrayValueErrorException_methods[] = {
        PHP_FE_END
};
static const zend_function_entry phpsci_ce_CArrayTypeErrorException_methods[] = {
        PHP_FE_END
};

static const zend_function_entry phpsci_ce_CArrayOverflowException_methods[] = {
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
    INIT_CLASS_ENTRY(ce, "CArrayValueErrorException", phpsci_ce_CArrayAxisException_methods);
    phpsci_ce_CArrayValueErrorException = zend_register_internal_class_ex(&ce, zend_ce_exception);
    INIT_CLASS_ENTRY(ce, "CArrayTypeErrorException", phpsci_ce_CArrayAxisException_methods);
    phpsci_ce_CArrayTypeErrorException = zend_register_internal_class_ex(&ce, zend_ce_exception);
    INIT_CLASS_ENTRY(ce, "CArrayOverflowException", phpsci_ce_CArrayOverflowException_methods);
    phpsci_ce_CArrayTypeErrorException = zend_register_internal_class_ex(&ce, zend_ce_exception);
}

/**
 * Throw CArrayAxisException
 */
void
throw_axis_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayAxisException, AXIS_EXCEPTION, "%s", msg);
}

/**
 * Throw ValueErrorException
 */
void
throw_valueerror_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayValueErrorException, VALUEERROR_EXCEPTION, "%s", msg);
}

/**
 * Throw TypeErrorException
 */
void
throw_typeerror_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayTypeErrorException, TYPEERROR_EXCEPTION, "%s", msg);
}

/**
 * Throw OverflowException
 * @param msg
 */
void
throw_overflow_exception(char * msg)
{
    zend_throw_exception_ex(phpsci_ce_CArrayOverflowException, OVERFLOW_EXCEPTION, "%s", msg);
}


