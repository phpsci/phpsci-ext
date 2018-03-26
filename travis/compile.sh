#!/bin/bash

phpize
./configure
make clean
make CFLAGS=-lopenblas
make install