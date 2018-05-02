--TEST--
basic test for CArray::inner();
--FILE--
<?php
$a = CArray::fromArray([1, 2, 3]);
$b = CArray::fromArray([4, 5, 6]);
$c = CArray::inner($a, $b);
echo $c;
?>
--EXPECT--
32.000000