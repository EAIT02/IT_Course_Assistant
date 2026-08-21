#!/bin/bash

Ns=(6 12 24 48) 
#Name=('Frcsub' 'Math1' 'Math2' 'ASSISTments2009' 'ASSISTments2017' 'Statics2011' 'Algebra2005')
Name=('Frcsub' 'Math1' 'Math2' 'ASSISTments2009')
N=6
Age1=10

for name in "${Name[@]}"
do
	python main.py "$name" $Age1 "$N" > results/ER-TGA-$name-Age$Age1-N$N-Psize30-Epsilon05-1-10.txt &
done

#for name in "${Name[@]}"
#do
#	for N in "${Ns[@]}"
#	do
#		python main.py "$name" $Age1 "$N" > results/ER-TGA-$name-Age$Age1-N$N.txt &
#	done
#done