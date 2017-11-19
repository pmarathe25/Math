BUILDDIR = build/
BINDIR = ~/bin/
TESTDIR = test/
SRCDIR = src/
# Objects
OBJS =
TESTOBJS = $(BUILDDIR)/MathTest.o
# Headers
INCLUDEPATH = include/
INCLUDE = -I$(INCLUDEPATH)
HEADERS = $(addprefix $(INCLUDEPATH)/, StealthMatrixBase.hpp)
# Compiler settings
CXX = nvcc
CFLAGS = -arch=sm_35 -Xcompiler -fPIC -Wno-deprecated-gpu-targets -c -std=c++11 -D_GLIBCXX_USE_C99 $(INCLUDE)
LFLAGS = -shared -Wno-deprecated-gpu-targets
TESTLFLAGS = -Wno-deprecated-gpu-targets
EXECLFLAGS = -Wno-deprecated-gpu-targets

$(TESTDIR)/MathTest: $(TESTOBJS)
	$(CXX) $(TESTLFLAGS) $(TESTOBJS) -o $(TESTDIR)/MathTest

$(BUILDDIR)/MathTest.o: $(TESTDIR)/MathTest.cu $(HEADERS)
	$(CXX) $(CFLAGS) $(TESTDIR)/MathTest.cu -o $(BUILDDIR)/MathTest.o


clean:
	rm $(OBJS) $(TESTOBJS) $(TESTDIR)/MathTest

test: $(TESTDIR)/MathTest
	$(TESTDIR)/MathTest
