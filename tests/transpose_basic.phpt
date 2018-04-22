--TEST--
basic test for CArray::transpose()
--FILE--
<?php
$a = CArray::fromArray([[0,1],[2,3]]);
$c = CArray::transpose($a);
print_r(CArray::toArray($c));
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 2
        )

    [1] => Array
        (
            [0] => 1
            [1] => 3
        )

)
