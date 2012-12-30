#!/bin/bash 

let i=0

while [[ $i -lt 10000 ]]
do 
	./GenDataMp4 $i >& /dev/null
	result=`./mp4 vecSumA.txt vecSumResult.txt vecSumResult.txt| tail -1`
	if [[ $result != "All tests passed!" ]]  
	then 
		echo failed at num $i  result: $result;
		exit -1
	fi
	let x=$i%100
	if [[ $x -eq 0 ]] 
	then 
		echo success upto $i
	fi
	let i=$i+1
done 
