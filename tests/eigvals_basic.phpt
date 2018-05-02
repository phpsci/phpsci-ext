--TEST--
basic test for CArray::eigvals()
--FILE--
<?php
$a = CArray::fromArray([[1, 2], [3, 4]]);

$eig = CArray::eigvals($a);
print_r(CArray::toArray($eig));
?>
--EXPECT--
Array
(
    [0] => -0.37228132326901
    [1] => 5.372281323269
)