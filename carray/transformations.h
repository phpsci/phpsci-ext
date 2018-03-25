//
// Created by hborba on 23/03/18.
//

#ifndef PHPSCI_EXT_TRANSFORMATIONS_H
#define PHPSCI_EXT_TRANSFORMATIONS_H

#include "../kernel/memory_manager.h"

void transpose(MemoryPointer * new_ptr, MemoryPointer * target_ptr, int rows, int cols);
#endif //PHPSCI_EXT_TRANSFORMATIONS_H
