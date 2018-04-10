#!/bin/bash
set -e

phpize
./configure
make clean
make test
