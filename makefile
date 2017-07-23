BUILDDIR = build/
BINDIR = ~/bin/
TESTDIR = test/
SRCDIR = src/
# Objects
OBJS = $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(BUILDDIR)/Matrix.o
TESTOBJS = $(BUILDDIR)/MathTest.o
EXECOBJS = $(BUILDDIR)/MathParse.o
# Libs
LIBDIR = $(CURDIR)/lib/
LIBTEXT = ~/C++/Text/lib/libtext.so
LIBTEXTINCLUDEPATH = /home/pranav/C++/Text/include/
INCLUDEPATH = include/
INCLUDEDIR = -I$(INCLUDEPATH) -I$(LIBTEXTINCLUDEPATH)
# Compiler settings
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 $(INCLUDEDIR)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(LIBDIR)/libmatrix.so: $(BUILDDIR)/Matrix.o
	$(CXX) $(LFLAGS) $(BUILDDIR)/Matrix.o $(LIBTEXT) -o $(LIBDIR)/libmatrix.so

$(LIBDIR)/libmath.so: $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o
	$(CXX) $(LFLAGS) $(BUILDDIR)/MathParser.o $(BUILDDIR)/Math.o $(LIBTEXT) -o $(LIBDIR)/libmath.so

$(BINDIR)/MathParse: $(EXECOBJS) $(LIBDIR)/libmath.so
	$(CXX) $(EXECLFLAGS) $(EXECOBJS) $(LIBDIR)/libmath.so -o $(BINDIR)/MathParse

$(BUILDDIR)/MathParse.o: $(SRCDIR)/MathParse.cpp $(LIBDIR)/libmath.so
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParse.cpp -o $(BUILDDIR)/MathParse.o

$(TESTDIR)/MathTest: $(TESTOBJS) $(LIBDIR)/libmath.so $(LIBDIR)/libmatrix.so
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) $(LIBDIR)/libmath.so $(LIBDIR)/libmatrix.so -o $(TESTDIR)/MathTest

$(BUILDDIR)/MathTest.o: $(TESTDIR)/MathTest.cu $(INCLUDEPATH)/Matrix.hpp $(LIBDIR)/libmath.so
	$(CXX) $(CFLAGS) $(TESTDIR)/MathTest.cu -o $(BUILDDIR)/MathTest.o

$(BUILDDIR)/MathParser.o: $(SRCDIR)/MathParser.cpp $(INCLUDEPATH)/MathParser.hpp $(INCLUDEPATH)/Math.hpp \
	$(LIBTEXTINCLUDEPATH)/strmanip.hpp
	$(CXX) $(CFLAGS) $(SRCDIR)/MathParser.cpp -o $(BUILDDIR)/MathParser.o

$(BUILDDIR)/Math.o: $(INCLUDEPATH)/Math.hpp $(SRCDIR)/Math.cu
	$(CXX) $(CFLAGS) $(SRCDIR)/Math.cu -o $(BUILDDIR)/Math.o

$(BUILDDIR)/Matrix.o: $(INCLUDEPATH)/Matrix.hpp $(INCLUDEPATH)/Math.hpp $(SRCDIR)/Matrix/Matrix.cu $(SRCDIR)/Matrix/MatrixComputationFunctions.cpp \
	$(SRCDIR)/Matrix/MatrixCUDAFunctions.cu $(SRCDIR)/Matrix/MatrixModificationFunctions.cpp $(SRCDIR)/Matrix/MatrixAccessFunctions.cpp
	$(CXX) $(CFLAGS) $(SRCDIR)/Matrix/Matrix.cu -o $(BUILDDIR)/Matrix.o

clean:
	rm $(OBJS) $(TESTOBJS) $(LIBDIR)/libmath.so $(TESTDIR)/MathTest

test: $(TESTDIR)/MathTest
	$(TESTDIR)/MathTest

exec: $(BINDIR)/MathParse

libmatrix: $(LIBDIR)/libmatrix.so

libmath: $(LIBDIR)/libmath.so
