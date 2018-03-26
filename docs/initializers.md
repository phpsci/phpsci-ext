# CArray Initializers

Methods for CArray creation.

---

### fromArray

```php
public static function fromArray(array $arr);
```
Return `MemoryPointer` of the converted PHP Array.

##### Parameters

- `array` $arr - Target PHP `array`to convert to `CArray`

##### Return

- `stdClass` - MemoryPointer of new converted `CArray` from target `array`

---

### identity

```php
public static function identity(int $size);
```
Return `MemoryPointer` of the identity matrix with size `$size`

##### Parameters

- `int` $size - Size of target square CArray

##### Return

- `stdClass` - MemoryPointer with identity CArray memory location

---

### zeros

```php
public static function zeros(int $rows, int $cols);
```
Return `MemoryPointer` of the a CArray full of zeros with shape ($rows, $cols)

###### Parameters

- `int` $rows - Number of rows of target CArray
- `int` $cols - Number of columns of target CArray
###### Return

- `stdClass` - MemoryPointer with CArray full of zeros

---