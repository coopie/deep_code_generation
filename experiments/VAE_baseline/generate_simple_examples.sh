#!/bin/sh


seq 1 1 100 | parallel -j1 --progress --bar "python generate_code.py > simple_examples/{}.hs"
