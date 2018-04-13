PHP_ARG_ENABLE(phpsci, whether to enable PHPSci computing library,
[  --disable-phpsci          Disable PHPSci computing library], yes)

if test "$PHP_PHPSCI" != "no"; then
  AC_DEFINE([HAVE_PHPSCI],1 ,[whether to enable  PHPSci computing library])
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



CFLAGS="$CFLAGS -lopenblas -llapacke"

PHP_NEW_EXTENSION(phpsci,
	  phpsci.c \
	  kernel/carray.c \
	  kernel/exceptions.c \
	  kernel/memory_manager.c \
	  carray/initializers.c \
	  carray/linalg.c \
	  carray/ranges.c \
	  carray/basic_operations.c \
	  carray/random.c \
	  carray/arithmetic.c \
	  kernel/carray_printer.c \
	  carray/transformations.c,
	  $ext_shared,, -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1)
  PHP_INSTALL_HEADERS([ext/phpsci], [phpsci.h])
  PHP_SUBST(PHPSCI_SHARED_LIBADD)
fi


