PROGS= mp1 mp2 mp3 mp4 mp5
LD=g++
HEADERS = $(wildcard include/*.h) $(wildcard include/*.hpp)
SHARED_DEPS = $(HEADERS) Makefile

CFLAGS = -g -Os -Wall -Wextra -Iinclude/ -std=c99 -DCUDA_EMU
CXXFLAGS = -g -Os -Wall -Wextra -Iinclude/ -DCUDA_EMU 
LDFLAGS = -lboost_thread-mt -lboost_system-mt


all: $(PROGS) GenDataMP1 GenDataMP2 GenDataMP4 GenDataMP5


.SUFFIXES:
.PRECIOUS: %.o

%: %.o
	$(LD) $< -o $@ $(LDFLAGS)

%.o: %.c $(SHARED_DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cc $(SHARED_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp $(SHARED_DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *~ *.o core $(PROGS) *.exe

vecC.txt: GenDataMP1
	./GenDataMP1 90

matC.txt: GenDataMP2
	./GenDataMP2 90 10 39 

vecSumResult.txt: GenDataMP4
	./GenDataMP4 1000 

vecCumSumResult.txt: GenDataMP5
	./GenDataMP5 100

run1: mp1 vecC.txt
	./mp1  vecA.txt vecB.txt vecC.txt
run2: mp2 matC.txt
	./mp2 matA.txt matB.txt matC.txt
run3: mp3 matC.txt
	./mp3 matA.txt matB.txt matC.txt

run4: mp4 vecSumResult.txt
	./mp4 vecSumA.txt vecSumResult.txt vecSumResult.txt 

run5: mp5 vecCumSumResult.txt
	./mp5 vecCumSumA.txt vecCumSumResult.txt vecCumSumResult.txt 

run: run1 run2 run3 run4 run5

# Unfortunately, Debian's astyle 2.01 doesn't support --align-reference=name
# This should be included when version 2.02 is used.
reformat:
	astyle \
		--style=stroustrup --align-pointer=name --indent-preprocessor \
		--pad-oper --pad-header --unpad-paren --indent=tab=8 \
		--min-conditional-indent=0 --max-instatement-indent=79 \
		--suffix=none \
		include/*.hpp
