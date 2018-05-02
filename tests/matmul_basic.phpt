--TEST--
basic test for CArray::matmul()
--FILE--
<?php
$a = CArray::fromArray([[1, 2, 3], [4, 5, 6]]);
$b = CArray::fromArray([[7, 8], [9, 10], [11, 12]]);
$c = CArray::matmul($a, $b);

print_r(CArray::toArray($c));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 58
            [1] => 64
        )

    [1] => Array
        (
            [0] => 139
            [1] => 154
        )

)