PROG1= mp1
PROG2= mp2
PROG3= mp3
HEADERS = $(wildcard include/*.h) $(wildcard include/*.hpp) Makefile
SOURCES1 = $(PROG1).cc $(HEADERS)
SOURCES2 = $(PROG2).cc $(HEADERS)
SOURCES3 = $(PROG3).cc $(HEADERS)

CFLAGS = -g -Os -Wall -Wextra -Iinclude/ -std=c99 -DCUDA_EMU
CXXFLAGS = -g -Os -Wall -Wextra -Iinclude/ -DCUDA_EMU 
LDFLAGS = -lboost_thread-mt -lboost_system-mt -lrt


all: mp1 GenDataMP1 GenDataMP2 

$(PROG1): $(SOURCES1)
	$(CXX) $(CXXFLAGS) $@.cc -o $@ $(LDFLAGS)
$(PROG2): $(SOURCES2)
	$(CXX) $(CXXFLAGS) $@.cc -o $@ $(LDFLAGS)
$(PROG3): $(SOURCES3)
	$(CXX) $(CXXFLAGS) $@.cc -o $@ $(LDFLAGS)

run: $(PROG)
	./$(PROG) foo foo

clean:
	rm -f *~ *.o core $(PROG1) $(PROG2) $(PROG3) *.exe

vecC.txt: GenDataMp1
	./GenDataMp1 90

matC.txt: GenDataMp2

run1: mp1 vecC.txt
	./mp1  vecA.txt vecB.txt vecC.txt
run2: mp2 matC.txt
	./mp2 matA.txt matB.txt matC.txt
run3: mp3 matC.txt
	./mp3 matA.txt matB.txt matC.txt


run:run1 run2 run3
