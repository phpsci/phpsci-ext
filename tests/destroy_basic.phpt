--TEST--
basic test for CArray::__destruct()
--FILE--
<?php
function create_scoped()
{
    $before = memory_get_usage(true);
    $a = CArray::identity(1024);
    var_dump(memory_get_usage(true) - $before == 1 << 22);
}

$before = memory_get_usage(true);
create_scoped();
gc_collect_cycles();
var_dump(memory_get_usage(true) == $before);
--EXPECT--
bool(true)
bool(true)