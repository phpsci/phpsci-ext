--TEST--
basic test for CArray::destroy()
--FILE--
<?php
$a = CArray::fromArray([[0,1],[2,3]]);
CArray::destroy($a->uuid, 2, 2);
?>
--EXPECT--
