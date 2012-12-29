PROGS= mp1 mp2 mp3
LD=g++
HEADERS = $(wildcard include/*.h) $(wildcard include/*.hpp)
SHARED_DEPS = $(HEADERS) Makefile

CFLAGS = -g -Os -Wall -Wextra -Iinclude/ -std=c99 -DCUDA_EMU
CXXFLAGS = -g -Os -Wall -Wextra -Iinclude/ -DCUDA_EMU 
LDFLAGS = -lboost_thread-mt -lboost_system-mt -lrt -stdc++


all: mp1 GenDataMP1 GenDataMP2 

.SUFFIXES:
.PRECIOUS: %.o

%: %.o
	$(LD) $< -o $@ $(LDFLAGS)

%.o: %.c $(SHARED_DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc $(SHARED_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *~ *.o core $(PROGS) *.exe

vecC.txt: GenDataMP1
	./GenDataMP1 90

matC.txt: GenDataMP2

run1: mp1 vecC.txt
	./mp1  vecA.txt vecB.txt vecC.txt
run2: mp2 matC.txt
	./mp2 matA.txt matB.txt matC.txt
run3: mp3 matC.txt
	./mp3 matA.txt matB.txt matC.txt

run: run1 run2 run2

# Unfortunately, Debian's astyle 2.01 doesn't support --align-reference=name
# This should be included when version 2.02 is used.
reformat:
	astyle \
		--style=stroustrup --align-pointer=name --indent-preprocessor \
		--pad-oper --pad-header --unpad-paren --indent=tab=8 \
		--min-conditional-indent=0 --max-instatement-indent=79 \
		--suffix=none \
		include/*.hpp
