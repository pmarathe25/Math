BUILDDIR = build/
INCLUDEDIR = include/
OBJS = $(BUILDDIR)/mathParser.o $(BUILDDIR)/math.o
TESTOBJS = $(BUILDDIR)/mathDemo.o
LIBDIR = lib/
LIBS = Text/libtext.so
TESTDIR = test/
SRCDIR = src/
CXX = g++
CFLAGS = -fPIC -c -std=c++11 -I./$(INCLUDEDIR)
LFLAGS = -shared
TESTLFLAGS =

libmath.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) $(LIBDIR)$(LIBS) -o libmath.so

$(TESTDIR)/MathDemo: $(OBJS) $(TESTOBJS)
	$(CXX) $(TESTLFLAGS) $(OBJS) $(TESTOBJS) $(LIBDIR)$(LIBS) -o $(TESTDIR)/MathDemo

$(BUILDDIR)/mathParser.o: $(INCLUDEDIR)/mathParser.hpp $(SRCDIR)/mathParser.cpp $(INCLUDEDIR)/math.hpp $(INCLUDEDIR)/Text/strmanip.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/mathParser.cpp -o $(BUILDDIR)/mathParser.o

$(BUILDDIR)/math.o: $(INCLUDEDIR)/math.hpp $(SRCDIR)/math.cpp
	$(CXX) $(CFLAGS) $(SRCDIR)/math.cpp -o $(BUILDDIR)/math.o

$(BUILDDIR)/mathDemo.o: $(TESTDIR)/mathDemo.cpp $(INCLUDEDIR)/mathParser.hpp $(INCLUDEDIR)/math.hpp
	$(CXX) $(CFLAGS) $(TESTDIR)/mathDemo.cpp -o $(BUILDDIR)/mathDemo.o

clean:
	rm $(OBJS) $(TESTOBJS)

test: $(TESTDIR)/MathDemo
	$(TESTDIR)/MathDemo
