# PHP Array and CArrays

Useful methods to work with other libraries that don't support
CArrays.

---

### toArray

```php
public static function toArray(CArray $obj);
```
Convert CArray to regular PHP Array.

##### Parameters

- `CArray` $obj - Target CArray

##### Return

- `array` - PHP Array with same shape and values of target CArray

---

### toDouble

```php
public static function toDouble(int $uuid);
```
Convert CArray (0D only) to double.

##### Parameters

- `int` $uuid - Memory Pointer of target CArray

##### Return

- `double` - Double representation of CArray 0D