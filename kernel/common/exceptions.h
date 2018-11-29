#ifndef PHPSCI_EXT_EXCEPTIONS_H
#define PHPSCI_EXT_EXCEPTIONS_H

#define AXIS_EXCEPTION 5000

void init_exception_objects();
void throw_axis_exception(char * msg);

#endif //PHPSCI_EXT_EXCEPTIONS_H