PHP_ARG_ENABLE(carray, whether to enable CArray computing library,
[  --disable-carray          Disable CArray computing library], yes)

if test "$PHP_CARRAY" != "no"; then
  AC_DEFINE([HAVE_CARRAY],1 ,[whether to enable  CArray computing library])
  AC_HEADER_STDC

AC_CHECK_HEADERS(
    [/usr/include/python3.5m/Python.h],
    [
        PHP_ADD_INCLUDE(/usr/include/python3.5m)
    ],
    ,
    [[#include "/usr/include/python3.5m/Python.h"]]
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
    [/usr/include/cblas.h],
    [
        PHP_ADD_INCLUDE(/usr/include/)
    ],
    ,
    [[#include "/usr/include/cblas.h"]]
)
AC_CHECK_HEADERS(
    [/usr/include/atlas/cblas.h],
    [
        PHP_ADD_INCLUDE(/usr/include/atlas/)
    ],
    ,
    [[#include "/usr/include/atlas/cblas.h"]]
)
AC_CHECK_HEADERS(
    [/usr/include/openblas/cblas.h],
    [
        PHP_ADD_INCLUDE(/usr/include/openblas/)
    ],
    ,
    [[#include "/usr/include/openblas/cblas.h"]]
)

PHP_CHECK_LIBRARY(blas,cblas_sdot,
[
  PHP_ADD_LIBRARY(blas)
],[
  PHP_CHECK_LIBRARY(openblas,cblas_sdot,
  [
    PHP_ADD_LIBRARY(openblas)
  ],[
    AC_MSG_ERROR([wrong openblas/blas version or library not found])
  ],[
    -lopenblas
  ])
],[
  -lblas
])

PHP_CHECK_LIBRARY(lapacke,LAPACKE_sgetrf,
[
  PHP_ADD_LIBRARY(lapacke)
],[
  AC_MSG_ERROR([wrong lapacke version or library not found])
],[
  -llapacke
])

PHP_CHECK_LIBRARY(python3.5m,Py_Initialize,
[
  PHP_ADD_LIBRARY(python3.5m)
],[
  AC_MSG_ERROR([wrong python version or library not found])
],[
 -lpython3.5m
])

CFLAGS="$CFLAGS -lopenblas -llapacke -lblas -llapack -lpython3.5m"

PHP_NEW_EXTENSION(carray,
	  phpsci.c \
	  src/buffer.c \
	  src/numeric.c \
	  src/exceptions.c \
	  src/linalg.c \
	  src/statistics.c \
      src/carray.c,
	  $ext_shared,, )
  PHP_INSTALL_HEADERS([ext/carray], [phpsci.h])
  PHP_SUBST(CARRAY_SHARED_LIBADD)
fi


