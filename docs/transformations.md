# Transforming CArrays

---

### transpose

```php
public static function transpose(int $a);
```
Transpose target CArray and returns it new `MemoryPointer`

> Remember it returns a NEW position in memory (read you need to destroy the old CArray).

##### Parameters

- `CArray` $a - target CArray

##### Return

- `CArray` - MemoryPointer of new transposed target array


---


### toArray

```php
public static function toArray($a);
```
Convert CArray to regular PHP Array.

##### Parameters

- `CArray` $a - target CArray

##### Return

- `array` - PHP Array with same shape em values of target CArray


