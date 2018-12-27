#ifndef PHPSCI_EXT_EXCEPTIONS_H
#define PHPSCI_EXT_EXCEPTIONS_H

#define AXIS_EXCEPTION       5000
#define VALUEERROR_EXCEPTION 5001
#define TYPEERROR_EXCEPTION  5002

void init_exception_objects();
void throw_axis_exception(char * msg);
void throw_valueerror_exception(char * msg);
void throw_typeerror_exception(char * msg);

#endif //PHPSCI_EXT_EXCEPTIONS_H