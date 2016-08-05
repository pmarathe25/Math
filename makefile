BUILDDIR = build/
INCLUDEDIR = include/
OBJS = $(BUILDDIR)/mathParser.o $(BUILDDIR)/math.o
TESTOBJS = $(BUILDDIR)/mathDemo.o
LIBDIR = lib/
LIBS = Text/libtext.so
TESTDIR = test/
SRCDIR = src/
EXECLFLAGS =
CXX = g++
CFLAGS = -fPIC -c -std=c++11 -I$(INCLUDEDIR)
LFLAGS = -shared

libmath.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) $(LIBDIR)/$(LIBS) -o libmath.so

$(TESTDIR)/MathDemo: $(OBJS) $(TESTOBJS)
	$(CXX) $(EXECLFLAGS) $(OBJS) $(TESTOBJS) $(LIBDIR)$(LIBS) -o $(TESTDIR)/MathDemo

$(BUILDDIR)/mathParser.o: $(SRCDIR)/mathParser.cpp $(INCLUDEDIR)/Math/mathParser.hpp $(INCLUDEDIR)/Math/math.hpp $(INCLUDEDIR)/Text/strmanip.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/mathParser.cpp -o $(BUILDDIR)/mathParser.o

$(BUILDDIR)/math.o: $(INCLUDEDIR)/Math/math.hpp $(SRCDIR)/math.cpp
	$(CXX) $(CFLAGS) $(SRCDIR)/math.cpp -o $(BUILDDIR)/math.o

$(BUILDDIR)/mathDemo.o: $(TESTDIR)/mathDemo.cpp $(INCLUDEDIR)/Math/mathParser.hpp $(INCLUDEDIR)/Math/math.hpp
	$(CXX) $(CFLAGS) $(TESTDIR)/mathDemo.cpp -o $(BUILDDIR)/mathDemo.o

clean:
	rm $(OBJS) $(TESTOBJS)

test: $(TESTDIR)/MathDemo
	$(TESTDIR)/MathDemo
