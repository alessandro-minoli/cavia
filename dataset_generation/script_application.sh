#!/bin/bash

echo ... generating applications with path topology

python3 gen_application.py 3  path 1 3
python3 gen_application.py 4  path 1 3
python3 gen_application.py 5  path 1 3
python3 gen_application.py 6  path 1 3
python3 gen_application.py 7  path 1 3
python3 gen_application.py 8  path 1 3
python3 gen_application.py 9  path 1 3
python3 gen_application.py 10 path 1 3

echo ... generating applications with tree topology

python3 gen_application.py 3  tree 3 3
python3 gen_application.py 4  tree 3 3
python3 gen_application.py 5  tree 3 3
python3 gen_application.py 6  tree 3 3
python3 gen_application.py 7  tree 3 3
python3 gen_application.py 8  tree 3 3
python3 gen_application.py 9  tree 3 3
python3 gen_application.py 10 tree 3 3

echo ... generating applications with graph topology

python3 gen_application.py 3  graph 3 3
python3 gen_application.py 4  graph 3 3
python3 gen_application.py 5  graph 3 3
python3 gen_application.py 6  graph 3 3
python3 gen_application.py 7  graph 3 3
python3 gen_application.py 8  graph 3 3
python3 gen_application.py 9  graph 3 3
python3 gen_application.py 10 graph 3 3