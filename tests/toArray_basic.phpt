--TEST--
basic test for CArray::toArray()
--FILE--
<?php
$a = CArray::identity(4);
$php_array = CArray::toArray($a);
print_r($php_array);
?>
--EXPECT--
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
