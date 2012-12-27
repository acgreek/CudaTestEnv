PROG = mp1

HEADERS = include/wb.h include/thread_processor.hpp
SOURCES = $(PROG).cc $(HEADERS)

CFLAGS = -g -Os -Wall -Wextra -Iinclude/ -std=c99 -DCUDA_EMU
CXXFLAGS = -g -Os -Wall -Wextra -Iinclude/ -DCUDA_EMU -pie
LDFLAGS = -lboost_thread-mt

$(PROG): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(PROG).cc -o $(PROG) $(LDFLAGS)


run: $(PROG)
	./$(PROG) foo foo

clean:
	rm -f *~ *.o core $(PROG)
