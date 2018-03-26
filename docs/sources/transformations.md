# Transforming CArrays

---

### transpose

```php
public static function transpose(int $uuid, int $rows, int $cols);
```
Transpose target CArray with shape (`$rows`, `$cols`) and returns
it new `MemoryPointer`

> Remember it returns a NEW position in memory (read you need to destroy the old CArray).

##### Parameters

- `int` $uuid - Memory Pointer of target CArray
- `int` $rows - Number of rows in target CArray
- `int` $cols - Number of cols in target CArray

##### Return

- `stdClass` - MemoryPointer of new transposed target array


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


