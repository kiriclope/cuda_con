#!/usr/bin/env bash
LANG=en_US
export LANG


for Cei in $(seq 1.0 0.1 1.5); do 
    cpt=0 ; 
    for Cie in $(seq 1.0 0.1 1.0); do 

	sed -i 's/IF_Dij .*/IF_Dij 1 /' devHostConstants.h 
	sed -i "s/Dij\[16\] .*/Dij\[16\]\ \=\{1.0,$Cei,$Cie,1.0,1.0,1.0,1.0,1.0,1.0,1.,1.,1.,1.,1.,1.,1.\}\;/" devHostConstants.h 
	sleep 5 
	echo "make Cie${Cie}Cei${Cei}"
	make 
	sleep 5 
	cpt=$((cpt+1))
	
	echo Cie${Cie}Cei${Cei} "Running ..." 
	screen -dmS Cie${Cie}Cei${Cei} ./a.out 	
	sleep 60 

	# if [ "$cpt" -gt "1" ]; then
	#     cpt=0 ;
	#     sleep 60 
	# fi
    done
    
done
