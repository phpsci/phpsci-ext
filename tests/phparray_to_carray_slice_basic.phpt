--TEST--
Basic test for 2-dimensional CArray slice with index
--FILE--
<?php
$a = new CArray([[1, 2], [3, 4]]);
$b = $a[0];
$b->print();
--EXPECT--
[ 1  2 ]