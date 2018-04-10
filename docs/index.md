# PHPSci CArray Extension

This is the extension used by [PHPSci](https://www.github.com/phpsci/phpsci). 
It offers the CArray object in place of PHP arrays to make scientific calculations faster.


Although it is not necessary, or you want to create your own library of arrays, we recommend that you use
[PHPSci](https://www.github.com/phpsci/phpsci) together with this extension.

> **ATTENTION:** Misuse of this extension can cause excessive memory consumption and consequently system crash. See the CArray internals section if you want to know more about the internal operation of the extension.

---

## Building

It's really easy to compile this extension using Linux environments.

#### Requirements

- php-devel (php-dev)
- PHP 7.x
- OpenBLAS

#### Compiling

Clone the repository, `cd` to the source folder and:
```commandline
$ phpize
$ ./configure
$ make test
$ make install
```
> Don't forget to check if the extension is enabled in your php.ini file.

> **Apache/NGINX Users:** Don't forget to restart your services.

---

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
CArray Object
(
    [uuid] => 0
)
CArray Object
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

---

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
`CArray Object` you see returned during use and contains the `uuid` property with
the position of your `CArray` inside the `MemoryStack`

So, when you do operations like `transpose`, the operation itself is only performed
with `C` objects and absolutely no PHP arrays are involved in the process.

That's what makes PHPSci so much faster them PHP Arrays.
    