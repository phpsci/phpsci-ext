--TEST--
basic test for CArray::inner() - 1D x 1D
--FILE--
<?php
$a = CArray::fromArray([1, 2, 3]);
$b = CArray::fromArray([4, 5, 6]);
$c = CArray::inner($a, $b);
echo $c;
?>
--EXPECT--
32.000000