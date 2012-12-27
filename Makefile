PROG = mp1

CFLAGS = -g -Os -Wall -Wextra -Iinclude/ -std=c99 -DCUDA_EMU
CXXFLAGS = -g -Os -Wall -Wextra -Iinclude/ -DCUDA_EMU -pie
LDFLAGS = -lboost_thread-mt

$(PROG): $(PROG).cc include/wb.h include/thread_processor.hpp
	$(CXX) $(CXXFLAGS) $(PROG).cc -o $(PROG) $(LDFLAGS)


run: $(PROG)
	./$(PROG) foo foo

clean:
	rm -f *~ *.o core $(PROG)
