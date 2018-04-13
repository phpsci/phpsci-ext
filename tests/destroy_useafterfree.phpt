--TEST--
Use-after-free bug demo for CArray::destroy()
--FILE--
<?php
$a = CArray::fromArray([[0,1],[2,3]]);
CArray::destroy($a);
// should print nothing here
?>
--EXPECT--