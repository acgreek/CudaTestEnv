#!/bin/bash 

let i=0

while [[ $i -lt 10000 ]]
do 
	GenDataMp4 $i
	result=`./mp4 vecSumA.txt vecSumResult.txt vecSumResult.txt| tail -1`
	if [[ $result != "All tests passed!" ]]  
	then 
		echo failed at $i;
		exit -1
	fi
	
done 
