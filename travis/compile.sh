#!/bin/bash

phpize
./configure
make clean
make
make install