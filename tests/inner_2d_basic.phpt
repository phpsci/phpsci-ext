--TEST--
basic test for CArray::inner() - 2D x 2D
--FILE--
<?php
$a = CArray::fromArray([[1, 2, 3],[4, 5, 6]]);
$b = CArray::fromArray([[1, 2, 3],[4, 5, 6]]);
$c = CArray::inner($a, $b);
print_r(CArray::toArray($c));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 14
            [1] => 32
        )

    [1] => Array
        (
            [0] => 32
            [1] => 77
        )

)