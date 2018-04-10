#!/bin/bash
set -e

phpize
./configure
make clean
make CFLAGS=-lopenblas
make test
