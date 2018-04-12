--TEST--
basic test for CArray::identity()
--FILE--
<?php
$a = CArray::identity(2);
$b = CArray::identity(4);
print_r(CArray::toArray($a));
print_r(CArray::toArray($b));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 1
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 0
            [2] => 0
            [3] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 1
            [2] => 0
            [3] => 0
        )

    [2] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 1
            [3] => 0
        )

    [3] => Array
        (
            [0] => 0
            [1] => 0
            [2] => 0
            [3] => 1
        )

)