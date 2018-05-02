--TEST--
basic test for CArray::eig()
--FILE--
<?php
$a = CArray::fromArray([[1, 2], [3, 4]]);

list($eig, $eig_v) = CArray::eig($a);
print_r(CArray::toArray($eig));
print_r(CArray::toArray($eig_v));
?>
--EXPECT--
Array
(
    [0] => -0.37228132326901
    [1] => 5.372281323269
)
Array
(
    [0] => Array
        (
            [0] => -0.82456484013239
            [1] => -0.41597355791928
        )

    [1] => Array
        (
            [0] => 0.56576746496899
            [1] => -0.90937670913212
        )

)