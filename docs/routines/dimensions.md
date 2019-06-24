# Dimensions Routines

---

## atleast_1d
```php
public static atleast_1d($a ...) : CArray
```
> Convert inputs to arrays with at least one dimension.
> Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.

##### Parameters

`CArray|Array` **$a ...** One or more input arrays.

##### Returns

`CArray|Array` An CArray, or array of CArrays, each with NDIM >= 1. Copies are made only if necessary.

---

## atleast_2d


## atleast_3d

## squeeze