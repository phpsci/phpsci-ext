# Random CArrays

---


### standard_normal
```php
public static function standard_normal(int $seed, int $x, int $y);
```
CArray filled with samples from a standard Normal distribution (mean=0, stdev=1).

##### Parameters

- `int` $seed -   Random seed number
- `int` $x - Number of rows if 2D matrix, width if 1D.
- `int` $y - Number of cols in 2D Matrix, 0 if 1D.

##### Return

- `CArray` - Returns CArray with shape (x,y) filled with random samples from
a standard normal distribution. 

