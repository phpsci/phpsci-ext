--TEST--
Basic test for identity matrix using CArray::identity
--FILE--
<?php
$a = CArray::identity(2);
$a->print();
--EXPECT--
[[ 1  0 ][ 0  1 ]]