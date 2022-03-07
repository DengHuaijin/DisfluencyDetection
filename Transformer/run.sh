#!/bin/bash

detection="negative"

./train.sh --detection $detection --num_set D
exit
./train.sh --detection $detection --num_set B
./train.sh --detection $detection --num_set C
./train.sh --detection $detection --num_set D
./train.sh --detection $detection --num_set E 
