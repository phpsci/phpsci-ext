--TEST--
basic test for CArray::zeros()
--FILE--
<?php
$a = CArray::zeros(2, 2);
$b = CArray::zeros(4, 3);
print_r(CArray::toArray($a->uuid, 2, 2));
print_r(CArray::toArray($b->uuid, 4, 3));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0
        )

)
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
        )

    [2] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
        )

    [3] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
        )

)