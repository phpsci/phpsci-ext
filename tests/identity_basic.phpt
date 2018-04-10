--TEST--
basic test for CArray::identity()
--FILE--
<?php
$a = CArray::identity(2);
$b = CArray::identity(4);
print_r($a);
print_r($b);
?>
--EXPECT--
CArray Object
(
    [uuid] => 0
    [x] => 2
    [y] => 2
)
CArray Object
(
    [uuid] => 1
    [x] => 4
    [y] => 4
)

