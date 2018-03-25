PHP_ARG_ENABLE(phpsci, whether to enable PHPSci computing library,
[  --disable-phpsci          Disable PHPSci computing library], yes)

if test "$PHP_PHPSCI" != "no"; then
  AC_DEFINE([HAVE_PHPSCI],1 ,[whether to enable  PHPSci computing library])
  AC_HEADER_STDC



PHP_NEW_EXTENSION(phpsci,
	  phpsci.c \
	  kernel/carray.c \
	  kernel/exceptions.c \
	  kernel/memory_manager.c \
	  carray/initializers.c \
	  carray/linalg.c \
	  carray/transformations.c,
	  $ext_shared,, -DZEND_ENABLE_STATIC_TSRMLS_CACHE=1)
  PHP_INSTALL_HEADERS([ext/phpsci], [phpsci.h])
  PHP_SUBST(PHPSCI_SHARED_LIBADD)
fi