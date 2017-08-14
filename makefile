BUILDDIR = build/
BINDIR = ~/bin/
TESTDIR = test/
SRCDIR = src/
# Objects
OBJS = $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(BUILDDIR)/StealthMatrix.o
TESTOBJS = $(BUILDDIR)/MathTest.o
EXECOBJS = $(BUILDDIR)/MathParse.o
# Libs
LIBDIR = $(CURDIR)/lib/
INCLUDEPATH = include/
INCLUDEDIR = -I$(INCLUDEPATH) -I/home/pranav/C++/Text/include/
# Compiler settings
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/libmatrix.so: $(BUILDDIR)/StealthMatrix.o
	$(CXX) $(LFLAGS) $(BUILDDIR)/StealthMatrix.o -o $(LIBDIR)/libmatrix.so

$(LIBDIR)/libmath.so: $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o
	$(CXX) $(LFLAGS) $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o -o $(LIBDIR)/libmath.so

$(BINDIR)/MathParse: $(EXECOBJS) $(LIBDIR)/libmath.so
	$(CXX) $(EXECLFLAGS) $(EXECOBJS) $(LIBDIR)/libmath.so -o $(BINDIR)/MathParse

$(BUILDDIR)/MathParse.o: $(SRCDIR)/MathParse.cpp $(LIBDIR)/libmath.so
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParse.cpp -o $(BUILDDIR)/MathParse.o

$(TESTDIR)/MathTest: $(TESTOBJS) $(LIBDIR)/libmath.so $(LIBDIR)/libmatrix.so
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBDIR)/libmath.so $(LIBDIR)/libmatrix.so -o $(TESTDIR)/MathTest

$(BUILDDIR)/MathTest.o: $(TESTDIR)/MathTest.cu $(INCLUDEPATH)/StealthMatrix.hpp $(LIBDIR)/libmath.so
	$(CXX) $(CFLAGS) $(TESTDIR)/MathTest.cu -o $(BUILDDIR)/MathTest.o

$(BUILDDIR)/MathParser.o: $(SRCDIR)/MathParser.cpp $(INCLUDEPATH)/MathParser.hpp $(INCLUDEPATH)/Math.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParser.cpp -o $(BUILDDIR)/MathParser.o

$(BUILDDIR)/Math.o: $(INCLUDEPATH)/Math.hpp $(SRCDIR)/Math.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/Math.cu -o $(BUILDDIR)/Math.o

$(BUILDDIR)/StealthMatrix.o: $(INCLUDEPATH)/StealthMatrix.hpp $(SRCDIR)/StealthMatrix/StealthMatrix.cu $(SRCDIR)/StealthMatrix/StealthMatrixComputationFunctions.cu \
	$(SRCDIR)/StealthMatrix/StealthMatrixCUDAFunctions.cu $(SRCDIR)/StealthMatrix/StealthMatrixModificationFunctions.cu $(SRCDIR)/StealthMatrix/StealthMatrixAccessFunctions.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/StealthMatrix/StealthMatrix.cu -o $(BUILDDIR)/StealthMatrix.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/libmath.so $(TESTDIR)/MathTest

test: $(TESTDIR)/MathTest
	$(TESTDIR)/MathTest

exec: $(BINDIR)/MathParse

libmatrix: $(LIBDIR)/libmatrix.so

libmath: $(LIBDIR)/libmath.so
