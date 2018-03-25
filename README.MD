# PHPSci CArray Extension

This is the extension used by [PHPSci](https://www.github.com/phpsci/phpsci). 
It offers the CArray object in place of PHP arrays to make scientific calculations faster.


Although it is not necessary, or you want to create your own library of arrays, we recommend that you use
[PHPSci](https://www.github.com/phpsci/phpsci) together with this extension.

> **ATTENTION:** Misuse of this extension can cause excessive memory consumption and consequently system crash. See the CArray internals section if you want to know more about the internal operation of the extension.

> **ATTENTION:** Memory allocation from this library dosen't talk with memory_limit directive, it WILL consume as much as you ask for.
## Building

It's really easy to compile this extension using Linux environments.

#### Requirements

- php-devel (php-dev)
- PHP 7.x

#### Compiling

Clone the repository, `cd` to the source folder and:
```php
$ phpize
$ ./configure
$ make clean
$ make
$ make install
```
> Don't forget to check if the extension is enabled in your php.ini file.

> **Apache/NGINX Users:** Don't forget to restart your services.

## Using CArrays

### Creating CArrays

Let's create two CArrays using the `Identity` initializer:

```php
$a = CArray::identity(2);
$b = CArray::identity(4);
print_r($a);
print_r($b);
```
```php
stdClass Object
(
    [uuid] => 0
)
stdClass Object
(
    [uuid] => 1
)
```
It sounds strange, but calm down! You will not be able to view your array using
`print_r` because CArrays are not PHP arrays. It's just pointers to memory, to view 
your array you'll need to convert it to PHP Array:

### Converting CArrays to PHP Arrays

Remember that this may require considerable time depending on the size of your CArray. Try performing all operations 
before converting to a PHP Array, and only, if needed of course.

```php
$php_array = CArray::toArray($a->uuid, 2, 2);
print_r($php_array);
```
> The `toArray()` static method receive 3 arguments: `public static toArray(int uuid, int rows, int cols);`. **For now, It's your
job to keep track of your array dimension and sizes. Misuse can cause segment faults.**

```php
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 0
        )

    [1] => Array
        (
            [0] => 0
            [1] => 1
        )

)
```
Now we can see our array and use it with other PHP general libraries.

### Creating from PHP Arrays

You also create CArrays from PHP Arrays, use the static `toArray()` method
to create a CArray from PHP Array:

```php
$a = CArray::fromArray([[0,1],[2,3]]);
print_r(CArray::toArray($a);
```
```php
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 1
        )

    [1] => Array
        (
            [0] => 2
            [1] => 3
        )

)
```

### Basic Operations

Let's tranpose the CArray (matrix) we created above:

```php
$c = CArray::transpose($a->uuid, 2,2);
print_r(CArray::toArray($c->uuid,2,2));
```
```php
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 2
        )

    [1] => Array
        (
            [0] => 1
            [1] => 3
        )

)
```
### Destroying CArrays [IMPORTANT]
After we are done with some CArrays or temporary ones, it's good to
destroy them by calling the `destroy()` static method.

```php
CArray::destroy($a->uuid);
CArray::destroy($b->uuid);
CArray::destroy($c->uuid);
```

> In small cases, this may not cause trouble, but in larger scales if you
don't destroy temporary CArrays, they will stay in memory until PHP runtime 
send the shutdown signal.

## How it works?
Internally CArrays are just C structures that can handle multiple arrays of
data.

```C                    
/*************          /*************** /***************
/*  CARRAY   *  ====>   /*   array1d   * /*  double[]   *
/*************          /*************** /***************
                        /*************** /*************** /************
                        /*   array2d   * /*   array1d   * /*  double  *
                        /*************** /*************** /************
                                     ...      
```
A buffer called `MemoryStack` handles all CArrays storage within your PC memory:
```php
/**********************           /*******************
/*   MEMORYSTACK      *           /   CArray UUID 0  *
/* Dynamic Allocated  *   =====>  /*******************
/*     Buffer         *           /*  CArray UUID 1  *
/**********************           /*******************
                                  /*  CArray UUID 2  *
                                  /*******************
                                                   ...
```
CArray talks with PHP frontend using only the `MemoryPointer` object, it's the
`stdClass` you see returned during use and contains the `uuid` property with
the position of your `CArray` inside the `MemoryStack`

So, when you do operations like `transpose`, the operation itself is only performed
with `C` objects and absolutely no PHP arrays are involved in the process.

That's what makes PHPSci so much faster them PHP Arrays.

# Methods

```php
public function toArray(int $uuid, int $rows, int $cols);
```
Return PHP `array` from `CArray` `MemoryPointer`

##### Parameters

- `int` $uuid - `MemoryPointer` UUID of target CArray to convert
- `int` $rows - Number of rows in target CArray
- `int` $cols - Number of columns in target CArray

##### Return

- `array` - Target `CArray` converted to PHP `array`

> Use this only when you really need PHP Arrays. (view, debug, etc..)


## Initializers
Useful tools for initialize a CArray:

### fromArray

```php
public static function fromArray(array $arr);
```
Return `MemoryPointer` of the converted PHP Array.

##### Parameters

- `array` $arr - Target PHP `array`to convert to `CArray`

##### Return

- `stdClass` - MemoryPointer of new converted `CArray` from target `array`

### Identity

```php
public static function identity(int $size);
```
Return `MemoryPointer` of the identity matrix with size `$size`

##### Parameters

- `int` $size - Size of target square CArray

##### Return

- `stdClass` - MemoryPointer with identity CArray memory location

### Zeros

```php
public static function zeros(int $rows, int $cols);
```
Return `MemoryPointer` of the a CArray full of zeros with shape ($rows, $cols)

##### Parameters

- `int` $rows - Number of rows of target CArray
- `int` $cols - Number of columns of target CArray
##### Return

- `stdClass` - MemoryPointer with CArray full of zeros

## Transformations
Methods for matrix transformation.

### Transpose

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

# Performance Comparison
Checkout the latest tests done so far:

### Matrix Creation
`i5 3550 3.4 GHz - 8GB RAM - Fedora 27`

Square identity with shape (x,x) using `microtime`:

PHP Array:

```php
PHP Array (1000, 1000): 0.13894987106323 secs
PHP Array (2000, 2000): 0.56144213676453 secs
PHP Array (3000, 3000): 1.2727909088135 secs
PHP Array (4000, 4000): 2.2695808410645 secs
PHP Array (5000, 5000): 3.5766291618347 secs
```

CArray:
```php
CArray (1000, 1000): 0.0042738914489746 secs
CArray (2000, 2000): 0.018796920776367 secs
CArray (3000, 3000): 0.03826117515564 secs
CArray (4000, 4000): 0.067836046218872 secs
CArray (5000, 5000): 0.1027500629425 secs
```

### Matrix Transpose
Considering matrix creation time.

PHP:

```php
PHP Array (1000, 1000): 0.33923411369324 secs
PHP Array (2000, 2000): 1.3257179260254 secs
PHP Array (3000, 3000): 3.0203940868378 secs
PHP Array (4000, 4000): 5.3613641262054 secs
PHP Array (5000, 5000): 9.7418010234833 secs
```

CArray:
```php
CArray (1000, 1000): 0.013009071350098 secs
CArray (2000, 2000): 0.062867164611816 secs
CArray (3000, 3000): 0.15480899810791 secs
CArray (4000, 4000): 0.27737402915955 secs
CArray (5000, 5000): 0.4391758441925 secs
```

### Matrix Product (matmul)
Not considering matrix creation time.

PHP:
```php
PHP Array (100, 100): 0.18620610237122 secs
PHP Array (200, 200): 1.4876849651337 secs
PHP Array (300, 300): 5.029287815094 secs
PHP Array (400, 400): 11.936593055725 secs
PHP Array (500, 500): 23.345940828323 secs
```

CArray:
```php
CArray (100, 100): 0.00039196014404297 secs
CArray (200, 200): 0.0028259754180908 secs
CArray (300, 300): 0.0093340873718262 secs
CArray (400, 400): 0.020502090454102 secs
CArray (500, 500): 0.040369987487793 secs
```

# Roadmap for version 1.0
Before release, we need to finish the following:

1. Implement `sum`, `abs`, `add`, `subtract`, `multiply`,`exp` and `std` operations
3. Implement `array1d` and `array0d` capabilities (allow 1-D and double operations and return)
4. Implement `zeros_like`, `ones`, `ones_like`, `linspace`, `eye`, `empty`, `fill` and `arange` initializers
5. Implement `ravel` and `flatten` transformations
6. Implement `matrix_power`, `inner`, `outer`, `kron` and `matmul` product operations
7. Implement `invert` from matrix inversion
