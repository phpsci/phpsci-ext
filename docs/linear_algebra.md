# Linear Algebra

---

### matmul
```php
public static function matmul(int $a_uuid, int $a_rows, int $a_cols, int $b_uuid, int $b_cols);
```
Matrix product of two CArrays.

- If both CArrays are 2-D they are multiplied like conventional matrices.
- If the first CArray is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
- If the second CArray is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
##### Parameters

- `int` $a_uuid - Memory Pointer of CArray A
- `int` $a_rows - Number of rows in target CArray A
- `int` $a_cols - Number of cols in target CArray A
- `int` $b_uuid - Memory Pointer of CArray B
- `int` $b_cols - Number of cols in target CArray B

##### Return

- `stdClass` - Returns the dot product of `A` and `B`.


