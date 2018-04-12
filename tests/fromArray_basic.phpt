--TEST--
basic test for CArray::fromArray()
--FILE--
<?php
$a = CArray::fromArray([[0, 1], [2, 3]]);
print_r(CArray::toArray($a));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 1
        )

    [1] => Array
        (
            [0] => 2
            [1] => 3
        )

)
