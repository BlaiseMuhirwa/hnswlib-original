#!/bin/bash 


set -ex 


rm -rf build 

mkdir -p build 

cd build 

cmake .. 

make 
