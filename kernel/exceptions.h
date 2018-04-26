/*
  +----------------------------------------------------------------------+
  | PHPSci CArray                                                        |
  +----------------------------------------------------------------------+
  | Copyright (c) 2018 PHPSci Team                                       |
  +----------------------------------------------------------------------+
  | Licensed under the Apache License, Version 2.0 (the "License");      |
  | you may not use this file except in compliance with the License.     |
  | You may obtain a copy of the License at                              |
  |                                                                      |
  |     http://www.apache.org/licenses/LICENSE-2.0                       |
  |                                                                      |
  | Unless required by applicable law or agreed to in writing, software  |
  | distributed under the License is distributed on an "AS IS" BASIS,    |
  | WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or      |
  | implied.                                                             |
  | See the License for the specific language governing permissions and  |
  | limitations under the License.                                       |
  +----------------------------------------------------------------------+
  | Authors: Henrique Borba <henrique.borba.dev@gmail.com>               |
  +----------------------------------------------------------------------+
*/

#ifndef PHPSCI_EXT_EXCEPTIONS_H
#define PHPSCI_EXT_EXCEPTIONS_H

/**
 * EXCEPTIONS CODES
 *
 * Format 5###
 */
#define COULD_NOT_BROADCAST_EXCEPTION 5000
#define SHAPES_NOT_ALIGNED_EXCEPTION  5001
#define ATLEAST2D_EXCEPTION           5002
#define OUTOFBOUNDS_EXCEPTION         5003

void init_exception_objects();

void throw_could_not_broadcast_exception(char * msg);
void throw_shapes_not_aligned_exception(char * msg);
void throw_atleast2d_exception(char * msg);
void throw_outofbounds_exception(char * msg);
#endif //PHPSCI_EXT_EXCEPTIONS_H
