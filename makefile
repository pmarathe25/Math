OBJS = mathParser.o math.o mathDemo.o ../Text/strmanip.o
CXX = g++
CFLAGS = -c -std=c++11
LFLAGS =

MathDemo: $(OBJS)
	$(CXX) $(LFLAGS) $(OBJS) -o MathDemo

mathParser.o: mathParser.hpp mathParser.cpp math.hpp ../Text/strmanip.hpp
	$(CXX) $(CFLAGS) mathParser.cpp

math.o: math.hpp math.cpp
	$(CXX) $(CFLAGS) math.cpp

mathDemo.o: mathDemo.cpp mathParser.hpp math.hpp
	$(CXX) $(CFLAGS) mathDemo.cpp

strmanip.o: ../Text/strmanip.hpp ../Text/strmanip.cpp
	$(CXX) $(CFLAGS) strmanip.cpp

clean:
	rm $(OBJS) MathDemo
