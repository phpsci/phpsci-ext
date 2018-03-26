# Performance Comparison




---

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

---

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

---

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