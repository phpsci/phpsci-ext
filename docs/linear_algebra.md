# Linear Algebra

---

### matmul
```php
public static function matmul(CArrat $a, CArray $b);
```
Matrix product of two CArrays.

- If both CArrays are 2-D they are multiplied like conventional matrices.
- If the first CArray is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
- If the second CArray is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
##### Parameters

- `CArray` $a_uuid - Array A
- `CArray` $b_uuid - Array B

##### Return

- `CArray` - Returns the dot product of `A` and `B`.


