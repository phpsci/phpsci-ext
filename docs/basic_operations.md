# Basic Operations

---

### sum

```php
public static function sum(int $uuid, int $x, int $y, int $axis);
```
Sum of CArray elements over a given axis.

###### Parameters

- `int` $uuid - Target CArray UUID
- `int` $x - If 2D, number of rows. If 1D, vector width.
- `int` $y - If 2D, number of columns. If 1D, set to 0.
- `int` $axis `OPTIONAL` - If null, will sum all elements of input CArray. 
###### Return

- `CArray` - CArray with sum result. If `axis = null` returns 0D.
---