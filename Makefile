CFLAGS=-g -Wall -I. -std=c99 -DCUDA_EMU
CXXFLAGS=-g -Wall -I. -DCUDA_EMU -pie 
LDFLAGS=-lboost_thread-mt -lboost_system-mt

all: mp1 GenDataMP1 GenDataMP2 

mp1: mp1.cc wb.h thread_processor.hpp
	g++ $(CXXFLAGS)  mp1.cc -o mp1 $(LDFLAGS)

mp2: mp2.cc wb.h
	g++ $(CXXFLAGS)  mp1.cc -o mp2 $(LDFLAGS)

mp3: mp3.cc wb.h
	g++ $(CXXFLAGS)  mp1.cc -o mp3 $(LDFLAGS)

vecC.txt: GenDataMp1
	./GenDataMp1 90

run1: mp1 vecC.txt
	./mp1  vecA.txt vecB.txt vecC.txt
run2: mp2
	./mp1 foo foo 
run3: mp3
	./mp1 foo foo 

run:run1 run2 run3
