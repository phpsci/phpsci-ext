--TEST--
basic test for CArray::destroy()
--FILE--
<?php
$a = CArray::fromArray([[100,1],[2,3]]);
var_dump($a->uuid);
CArray::destroy($a->uuid, 2, 2);
var_dump($a->uuid);

$b = CArray::fromArray([[200,1],[2,3]]);
var_dump($b->uuid);
CArray::destroy($b->uuid, 2, 2);
var_dump($b->uuid);
?>
--EXPECT--
int(0)
int(0)
int(0)
int(0)