--TEST--
Basic test for 3-dimensional CArray creation (_construct) using PHP arrays
--FILE--
<?php
$a = new CArray([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]);
--EXPECT--