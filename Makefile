PROG = mp1

CFLAGS=-g -Wall -I. -std=c99 -DCUDA_EMU
CXXFLAGS=-g -Wall -I. -DCUDA_EMU -pie
LDFLAGS=-lboost_thread-mt

$(PROG): $(PROG).cc wb.h thread_processor.hpp
	g++ $(CXXFLAGS) $(PROG).cc -o $(PROG) $(LDFLAGS)


run: $(PROG)
	./$(PROG) foo foo

