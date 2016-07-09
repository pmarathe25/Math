#include "mathParser.hpp"
#include <iostream>

int main() {
    MathParser parser = MathParser();
    std::cout << parser.parse("5+4") << std::endl;
    return 0;
}
