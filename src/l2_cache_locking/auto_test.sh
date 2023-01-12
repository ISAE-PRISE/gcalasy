# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# GCALASY project
# Copyright (C) 2023 ISAE
# 
# Purpose:
# Studying the response of cache memory interference to hardware-based 
# cache locking technique on iGPUs
#
# Contacts:
# alfonso.mascarenas-gonzalez@isae-supaero.fr
# jean-baptiste.chaudron@isae-supaero.fr
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

#! /bin/bash
# Automatized tests as function of memory usage and reserved locking capactiy of the L2 cache
# The shell argument indicates the test to perform:
# 0 = Matrix multiplication
# 1 = Matrix-vector multiplication-addition
# 2 = Matrix convolution 
# 3 = Matrix max-pool upsampling
# else = Vector data reset

echo "Executing cache locking tests: "

# Matrix multiplication
if [[ $1 == 0 ]]
then
	echo "Matrix multiplication benchmark"
	echo ""
	for matrix_size_multiplier in {15..29..1} # {15..29..1}
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) # seq 0 0.25 3 
		do
			echo "Lock size: $cache_lock_size"
			for iter in {0..99..1}
			do
				./cache_locking $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done


# Matrix-vector multiplication-addition
elif [[ $1 == 1 ]]
then
	echo "Matrix-vector multiplication-addition benchmark"
	echo ""
	for matrix_size_multiplier in {35..50..1} # {15..29..1}
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) # seq 0 0.25 3 
		do
			echo "Lock size: $cache_lock_size"
			for iter in {0..99..1}
			do
				./cache_locking $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done
	
	
# Matrix convolution
elif [[ $1 == 2 ]]
then
	echo "Matrix convolution benchmark"
	echo ""
	for matrix_size_multiplier in {5..14..1} # {20..34..1}
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) # seq 0 0.25 3 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..99..1}
			do
				./cache_locking $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done


# Matrix Nearest-Neighbor upsampling
elif [[ $1 == 3 ]]
then
	echo "Matrix upsampling benchmark"
	echo ""
	for matrix_size_multiplier in {20..35..1} # {20..34..1}
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) # seq 0 0.25 3 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..99..1}
			do
				./cache_locking $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done


# Reset vector	
else
	echo "Vector resetting benchmark"
	echo ""
	for vector_size_multiplier in {6..9..1} # {20..34..1}
	do
		echo "Vector size multiplier: $vector_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) # seq 0 0.25 3 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..10..1}
			do
				./cache_locking $cache_lock_size $1 $vector_size_multiplier		 
			done
		done
	done
fi






















