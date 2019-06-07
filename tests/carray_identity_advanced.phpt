--TEST--
Basic test for identity matrix using CArray `flat` property
--FILE--
<?php
$a = new CArray([[0,0,0],[0,0,0],[0,0,0]]);
$b = new CArray([1,0,0,0]);
$a->flat = $b;
$a->print();
--EXPECT--
[[ 1  0  0 ][ 0  1  0 ][ 0  0  1 ]]