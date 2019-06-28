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

CFLAGS="$CFLAGS -lopenblas -llapacke -lblas -llapack"

PHP_NEW_EXTENSION(carray,
	  phpsci.c \
	  kernel/alloc.c \
	  kernel/carray.c \
      kernel/iterators.c \
      kernel/flagsobject.c \
      kernel/assign.c \
      kernel/convert.c \
      kernel/casting.c \
      kernel/linalg.c \
      kernel/calculation.c \
      kernel/shape.c \
      kernel/common/common.c \
      kernel/common/cblas_funcs.c \
      kernel/common/mem_overlap.c \
      kernel/number.c \
      kernel/convert_type.c \
      kernel/trigonometric.c \
      kernel/matlib.c \
      kernel/statistics.c \
      kernel/arraytypes.c \
      kernel/join.c \
      kernel/ctors.c \
      kernel/scalar.c \
      kernel/getset.c \
      kernel/common/strided_loops.c \
      kernel/convert_datatype.c \
      kernel/dtype_transfer.c \
      kernel/assign_scalar.c \
      kernel/common/exceptions.c \
      kernel/item_selection.c \
      kernel/clip.c \
      kernel/search.c \
      kernel/common/sort.c \
      kernel/common/compare.c \
      kernel/exp_logs.c \
      kernel/random.c \
      kernel/range.c \
      kernel/conversion_utils.c \
	  kernel/buffer.c ,
	  $ext_shared,, )
  PHP_INSTALL_HEADERS([ext/carray], [phpsci.h, kernel/carray.h, kernel/types.h])
  PHP_SUBST(CARRAY_SHARED_LIBADD)
fi


