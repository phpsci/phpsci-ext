--TEST--
Advanced test for 3-dimensional CArray slice with multiple indexes
--FILE--
<?php
$a = new CArray([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]);
$b = $a[0][1];
$b->print();
--EXPECT--
[ 3  4 ]