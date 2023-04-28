#! /bin/bash
# Automatized tests as function of memory usage and reserved locking capactiy of the L2 cache
# The shell argument indicates the test to perform:
# 0 = Vector data reset under inter-SM interference with one application
# 1 = 2D convolution under inter-SM interference with one application
# 2 = Matrix max-pool upsampling under inter-SM interference with one application
# 10 = Vector data reset under inter-SM interference with several applications
# 11 = 2D convolution under inter-SM interference with several applications
# 12 = 3D convolution under inter-SM interference with several applications
# 20 = Vector data reset under DDR SDRAM and inter-SM interference with one application
# 21 = 2D convolution under DDR SDRAM and inter-SM interference with one application
# 22 = Matrix max-pool upsampling under DDR SDRAM and inter-SM interference with one application

echo "Executing cache locking tests: "


##########################################################
# Scenario 1: Inter-SM interference with one application #
##########################################################

# Data reset	
if [[ $1 == 0 ]]
then
	echo "Data reset benchmark"
	echo ""
	for matrix_size_multiplier in {1..9..1}  
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done	
	
	
# 2D convolution
elif [[ $1 == 1 ]]
then
	echo "2D convolution benchmark"
	echo ""
	for matrix_size_multiplier in {5..14..1} 
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done


# Matrix Nearest-Neighbor upsampling
elif [[ $1 == 2 ]]
then
	echo "Matrix upsampling benchmark"
	echo ""
	for matrix_size_multiplier in {20..35..1} 
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 1 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done
	
	
	
###############################################################
# Scenario 2: Inter-SM interference with several applications #
###############################################################
	
# Data reset (mix-criticality)
elif [[ $1 == 10 ]]
then
	echo "Data reset benchmark (mix-criticality)"
	echo ""
		for cache_lock_size in $(seq 0 3 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..0..1}
			do
				./main $cache_lock_size $1 3		 
			done
		done
	
	
# 2D convolution (mix-criticality)
elif [[ $1 == 11 ]]
then
	echo "2D benchmark (mix-criticality)"
	echo ""
		for cache_lock_size in $(seq 0 3 3)
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..0..1}
			do
				./main $cache_lock_size $1 1		 
			done
		done

		
# 3D convolution (mix-criticality)
elif [[ $1 == 12 ]]
then
	echo "3D convolution benchmark (mix-criticality)"
	echo ""
		for cache_lock_size in $(seq 0 3 3)
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..0..1}
			do
				./main $cache_lock_size $1 1		 
			done
		done



#########################################################################
# Scenario 3: DDR SDRAM and inter-SM interference with one application #
########################################################################

# Data reset 
elif [[ $1 == 20 ]]
then
	echo "Data reset benchmark with SDRAM interference"
	echo ""
	for matrix_size_multiplier in {1..9..1}  
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 3 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done	
	
	
# 2D convolution
elif [[ $1 == 21 ]]
then
	echo "2D convolution benchmark with SDRAM interference"
	echo ""
	for matrix_size_multiplier in {5..14..1} 
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 3 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done


# Matrix Nearest-Neighbor upsampling
elif [[ $1 == 22 ]]
then
	echo "Matrix upsampling benchmark with SDRAM interference"
	echo ""
	for matrix_size_multiplier in {20..35..1} 
	do
		echo "Matrix size multiplier: $matrix_size_multiplier"
		for cache_lock_size in $(seq 0 3 3) 
		do
			echo "Max lock size: $cache_lock_size"
			for iter in {0..999..1}
			do
				./main $cache_lock_size $1 $matrix_size_multiplier		 
			done
		done
	done	
	
				
else
	echo "Test does not exist"

fi






















