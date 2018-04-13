--TEST--
basic test for CArray::sum()
--FILE--
<?php
$carray = CArray::fromArray([[0, 1], [0, 5]]);
$result_y = CArray::sum($carray, 0);
print_r(CArray::toArray($result_y));
$result_x = CArray::sum($carray, 1);
print_r(CArray::toArray($result_x));
?>
--EXPECT--
Array
(
    [0] => 0
    [1] => 6
)
Array
(
    [0] => 1
    [1] => 5
)