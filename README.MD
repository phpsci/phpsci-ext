# PHPSci CArray Extension
[![Build Status](https://travis-ci.org/phpsci/phpsci-ext.svg?branch=master)](https://travis-ci.org/phpsci/phpsci-ext)
[![Documentation Status](https://readthedocs.org/projects/phpsci-carray/badge/?version=latest)](http://php-learn.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <img src="https://i.imgur.com/QoIbhqj.png" width="70%" />
</p>

This is the extension used by [PHPSci](https://www.github.com/phpsci/phpsci). 
It offers the CArray object in place of PHP arrays to make scientific calculations faster.


Although it is not necessary, or you want to create your own library of arrays, we recommend that you use
[PHPSci](https://www.github.com/phpsci/phpsci) together with this extension.


---

## Installing

It's really easy to compile this extension using Linux environments.

#### Requirements

- php-devel (php-dev)
- PHP 7.2
- OpenBLAS

#### Ubuntu 16.04
```commandline
$ add-apt-repository -y ppa:ondrej/php
$ apt-get update
$ apt-get install libblas-dev libatlas-base-dev php7.2-dev
$ phpize
$ ./configure
$ make test
$ make install
```
#### Ubuntu 14.04
```commandline
$ add-apt-repository -y ppa:ondrej/php
$ apt-get update
$ apt-get install libopenblas-dev liblapacke-dev php7.2-dev
$ phpize
$ ./configure
$ make test
$ make install
```

> Don't forget to check if the extension is enabled in your php.ini file.

> **Apache/NGINX Users:** Don't forget to restart your services.

---

