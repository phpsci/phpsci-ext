PHP_ARG_ENABLE(carray, whether to enable CArray computing library,
[  --disable-carray          Disable CArray computing library], yes)

if test "$PHP_CARRAY" != "no"; then
  AC_DEFINE([HAVE_CARRAY],1 ,[whether to enable  CArray computing library])
  AC_HEADER_STDC

AC_CHECK_HEADERS(
    [/opt/OpenBLAS/include/lapacke.h],
    [
        PHP_ADD_INCLUDE(/opt/OpenBLAS/include/)
    ],
    ,
    [[#include "/opt/OpenBLAS/include/lapacke.h"]]
)
AC_CHECK_HEADERS(
    [/usr/include/openblas/lapacke.h],
    [
        PHP_ADD_INCLUDE(/usr/include/openblas/)
    ],
    ,
    [[#include "/usr/include/openblas/lapacke.h"]]
)
AC_CHECK_HEADERS(
    [/usr/include/lapacke.h],
    [
        PHP_ADD_INCLUDE(/usr/include/)
    ],
    ,
    [[#include "/usr/include/lapacke.h"]]
)


AC_CHECK_HEADERS(
    [/opt/OpenBLAS/include/cblas.h],
    [
        PHP_ADD_INCLUDE(/opt/OpenBLAS/include/)
    ],
    ,
    [[#include "/opt/OpenBLAS/include/cblas.h"]]
)
AC_CHECK_HEADERS(
    [/usr/include/openblas/cblas.h],
    [
        PHP_ADD_INCLUDE(/usr/include/openblas/)
    ],
    ,
    [[#include "/usr/include/openblas/cblas.h"]]
)

PHP_CHECK_LIBRARY(openblas,cblas_sdot,
[
  PHP_ADD_LIBRARY(openblas)
],[
  AC_MSG_ERROR([wrong openblas version or library not found])
],[
  -lopenblas
])

PHP_CHECK_LIBRARY(lapacke,LAPACKE_sgetrf,
[
  PHP_ADD_LIBRARY(lapacke)
],[
  AC_MSG_ERROR([wrong lapacke version or library not found])
],[
  -llapacke
])

CFLAGS="$CFLAGS -lopenblas -llapacke"

PHP_NEW_EXTENSION(carray,
	  phpsci.c \
	  kernel/carray/carray.c \
	  kernel/carray/tuple.c \
	  kernel/exceptions.c \
	  kernel/memory_pointer/memory_pointer.c \
	  kernel/memory_pointer/utils.c \
	  kernel/buffer/memory_manager.c \
	  operations/initializers.c \
	  operations/linalg.c \
	  operations/ranges.c \
	  operations/basic_operations.c \
	  operations/random.c \
	  operations/arithmetic.c \
	  operations/exponents.c \
	  operations/logarithms.c \
	  operations/trigonometric.c \
	  operations/hyperbolic.c \
	  operations/transformations.c \
	  operations/magic_properties.c \
	  operations/linalg/norms.c \
	  operations/linalg/others.c \
	  operations/linalg/eigenvalues.c \
	  operations/linalg/equations.c \
	  kernel/carray/utils/carray_printer.c \
	  kernel/php/php_array.c ,
	  $ext_shared,, )
  PHP_INSTALL_HEADERS([ext/carray], [phpsci.h])
  PHP_SUBST(CARRAY_SHARED_LIBADD)
fi


