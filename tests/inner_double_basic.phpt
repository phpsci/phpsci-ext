--TEST--
basic test for CArray::inner() - $b Double
--FILE--
<?php
$a = CArray::fromArray([[1, 2, 3],[4, 5, 6]]);
$c = CArray::inner($a, 2.0);
print_r(CArray::toArray($c));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
            [2] => 6
        )

    [1] => Array
        (
            [0] => 8
            [1] => 10
            [2] => 12
        )

)