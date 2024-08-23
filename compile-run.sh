#!/bin/bash

nvcc -arch=sm_86 -gencode=arch=compute_86,code=sm_86 src/main.cu -o main
./main