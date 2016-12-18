BUILDDIR = build/
BINDIR = ~/bin/
INCLUDEDIR = include/
OBJS = $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o
TESTOBJS = $(BUILDDIR)/MathDemo.o
EXECOBJS = $(BUILDDIR)/MathParse.o
LIBDIR = $(CURDIR)/lib/
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS =  -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 -I$(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/Math/libmath.so: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) -o $(LIBDIR)/Math/libmath.so

$(BINDIR)/MathParse: $(EXECOBJS) $(LIBDIR)/Math/libmath.so
	$(CXX) $(EXECLFLAGS) $(EXECOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/Text/libtext.so -o $(BINDIR)/MathParse

$(BUILDDIR)/MathParse.o: $(SRCDIR)/MathParse.cpp $(LIBDIR)/Math/libmath.so
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParse.cpp -o $(BUILDDIR)/MathParse.o

$(TESTDIR)/MathDemo: $(TESTOBJS) $(LIBDIR)/Math/libmath.so
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/Text/libtext.so -o $(TESTDIR)/MathDemo

$(BUILDDIR)/MathDemo.o: $(TESTDIR)/MathDemo.cpp $(INCLUDEDIR)/Math/Matrix.hpp $(LIBDIR)/Math/libmath.so
	$(CXX) $(CFLAGS) $(TESTDIR)/MathDemo.cpp -o $(BUILDDIR)/MathDemo.o

$(BUILDDIR)/MathParser.o: $(SRCDIR)/MathParser.cpp $(INCLUDEDIR)/Math/MathParser.hpp $(INCLUDEDIR)/Math/Math.hpp $(INCLUDEDIR)/Text/strmanip.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParser.cpp -o $(BUILDDIR)/MathParser.o

$(BUILDDIR)/Math.o: $(INCLUDEDIR)/Math/Math.hpp $(SRCDIR)/Math.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/Math.cu -o $(BUILDDIR)/Math.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(TESTDIR)/MathDemo

test: $(TESTDIR)/MathDemo
	$(TESTDIR)/MathDemo

exec: $(BINDIR)/MathParse
