# Linear Algebra

---

### matmul
```php
public static function matmul(int $a_uuid, int $a_rows, int $a_cols, int $b_uuid, int $b_cols);
```
Matrix product of two arrays.

- If both CArrays are 2-D they are multiplied like conventional matrices.

##### Parameters

- `int` $a_uuid - Memory Pointer of CArray A
- `int` $a_rows - Number of rows in target CArray A
- `int` $a_cols - Number of cols in target CArray A
- `int` $b_uuid - Memory Pointer of CArray B
- `int` $b_cols - Number of cols in target CArray B

##### Return

- `stdClass` - Returns the dot product of `A` and `B`.


