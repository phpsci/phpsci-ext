--TEST--
Basic test for print after CArray creation (_construct) using PHP arrays
--FILE--
<?php
$a = new CArray([[1, 2], [3, 4]]);
$a->print();
--EXPECT--
[[ 1  2 ][ 3  4 ]]