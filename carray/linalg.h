//
// Created by hborba on 23/03/18.
//

#ifndef PHPSCI_EXT_LINALG_H
#define PHPSCI_EXT_LINALG_H


#include "../kernel/memory_manager.h"


void matmul(MemoryPointer * ptr, int n_a_rows, int n_a_cols, MemoryPointer * a_ptr, int n_b_cols, MemoryPointer *b_ptr);
#endif //PHPSCI_EXT_LINALG_H
