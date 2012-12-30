#!/bin/bash 

let x=1

mod_v=1000
while [[ $x -lt 10000 ]]
do 
	let y=1
	while [[ $y -lt 10000 ]]
	do 
		let z=0
		while [[ $z -lt 1000 ]]
		do 
			k=$RANDOM%1000
			j=$RANDOM%1000
			./GenDataMP2 $k $j $z >& /dev/null
			result=`./mp3 matA.txt matB.txt matC.txt| tail -1`
			if [[ $result != "All tests passed!" ]]  
			then 
				echo failed at $x $y $z  result: $result;
				exit -1
			fi
			echo success at $k $j $z
			r=$RANDOM%100
			let z=$z+$r;

		done 
		r=$RANDOM%$mod_v
		let y=$y+$r;
	done 
	r=$RANDOM%$mod_v
	let x=$x+$r;
done 
