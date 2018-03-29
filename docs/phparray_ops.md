# PHP Array and CArrays

Useful methods to work with other libraries that don't support
CArrays.

---

### toArray

```php
public static function toArray(int $uuid, int $rows, int $cols);
```
Convert CArray with shape (`$rows`, `$cols`) to regular
PHP Array.

##### Parameters

- `int` $uuid - Memory Pointer of target CArray
- `int` $rows - Number of rows in target CArray
- `int` $cols - Number of cols in target CArray

##### Return

- `array` - PHP Array with same shape em values of target CArray

---

### toDouble

```php
public static function toArray(int $uuid);
```
Convert CArray (0D only) to double.

##### Parameters

- `int` $uuid - Memory Pointer of target CArray

##### Return

- `double` - Double representation of CArray 0D