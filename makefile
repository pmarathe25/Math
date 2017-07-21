BUILDDIR = build/
BINDIR = ~/bin/
INCLUDEDIR = include/
OBJS = $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(BUILDDIR)/Matrix.o
TESTOBJS = $(BUILDDIR)/MathTest.o
EXECOBJS = $(BUILDDIR)/MathParse.o
LIBDIR = $(CURDIR)/lib/
TESTDIR = test/
SRCDIR = src/
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 -I$(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/Math/libmatrix.so: $(BUILDDIR)/Matrix.o $(LIBDIR)/Text/libtext.so
	$(CXX) $(LFLAGS) $(BUILDDIR)/Matrix.o $(LIBDIR)/Text/libtext.so -o $(LIBDIR)/Math/libmatrix.so

$(LIBDIR)/Math/libmath.so: $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(LIBDIR)/Text/libtext.so
	$(CXX) $(LFLAGS) $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(LIBDIR)/Text/libtext.so -o $(LIBDIR)/Math/libmath.so

$(BINDIR)/MathParse: $(EXECOBJS) $(LIBDIR)/Math/libmath.so
	$(CXX) $(EXECLFLAGS) $(EXECOBJS) $(LIBDIR)/Math/libmath.so -o $(BINDIR)/MathParse

$(BUILDDIR)/MathParse.o: $(SRCDIR)/MathParse.cpp $(LIBDIR)/Math/libmath.so
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParse.cpp -o $(BUILDDIR)/MathParse.o

$(TESTDIR)/MathTest: $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/Math/libmatrix.so
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(LIBDIR)/Math/libmatrix.so -o $(TESTDIR)/MathTest

$(BUILDDIR)/MathTest.o: $(TESTDIR)/MathTest.cu $(INCLUDEDIR)/Math/Matrix.hpp $(LIBDIR)/Math/libmath.so
	$(CXX) $(CFLAGS) $(TESTDIR)/MathTest.cu -o $(BUILDDIR)/MathTest.o

$(BUILDDIR)/MathParser.o: $(SRCDIR)/MathParser.cpp $(INCLUDEDIR)/Math/MathParser.hpp $(INCLUDEDIR)/Math/Math.hpp \
	$(INCLUDEDIR)/Text/strmanip.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParser.cpp -o $(BUILDDIR)/MathParser.o

$(BUILDDIR)/Math.o: $(INCLUDEDIR)/Math/Math.hpp $(SRCDIR)/Math.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/Math.cu -o $(BUILDDIR)/Math.o

$(BUILDDIR)/Matrix.o: $(INCLUDEDIR)/Math/Matrix.hpp $(INCLUDEDIR)/Math/Math.hpp $(SRCDIR)/Matrix/Matrix.cu $(SRCDIR)/Matrix/MatrixComputationFunctions.cpp \
	$(SRCDIR)/Matrix/MatrixCUDAFunctions.cu $(SRCDIR)/Matrix/MatrixModificationFunctions.cpp $(SRCDIR)/Matrix/MatrixAccessFunctions.cpp
	$(CXX) $(CFLAGS) $(SRCDIR)/Matrix/Matrix.cu -o $(BUILDDIR)/Matrix.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/Math/libmath.so $(TESTDIR)/MathTest

test: $(TESTDIR)/MathTest
	$(TESTDIR)/MathTest

exec: $(BINDIR)/MathParse

libmatrix: $(LIBDIR)/Math/libmatrix.so

libmath: $(LIBDIR)/Math/libmath.so
