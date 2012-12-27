CFLAGS=-g -Wall -I. -std=c99 -DCUDA_EMU
CXXFLAGS=-g -Wall -I. -DCUDA_EMU -pie
LDFLAGS=-lboost_thread-mt -lboost_system-mt

mp1: mp1.cc wb.h thread_processor.hpp
	g++ $(CXXFLAGS)  mp1.cc -o mp1 $(LDFLAGS)


run: mp1
	./mp1 foo foo
