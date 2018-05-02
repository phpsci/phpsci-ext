--TEST--
basic test for CArray::svd()
--FILE--
<?php
$a = CArray::fromArray([[1, 2], [3, 4]]);

list($un_a, $sv, $un_b) = CArray::svd($a);
print_r(CArray::toArray($un_a));
print_r(CArray::toArray($sv));
print_r(CArray::toArray($un_b));
?>
--EXPECT--
Array
(
    [0] => 5.464985704219
    [1] => 0.36596619062626
)
Array
(
    [0] => Array
        (
            [0] => -0.40455358483376
            [1] => -0.9145142956773
        )

    [1] => Array
        (
            [0] => -0.9145142956773
            [1] => 0.40455358483376
        )

)
Array
(
    [0] => Array
        (
            [0] => -0.57604843676632
            [1] => -0.81741556047036
        )

    [1] => Array
        (
            [0] => 0.81741556047036
            [1] => -0.57604843676632
        )

)