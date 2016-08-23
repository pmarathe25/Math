#include "Math/MathParser.hpp"
#include <string>
#include <iostream>

int main(int argc, char* argv[]) {
    math::MathParser parser = math::MathParser();
    std::string arg;
    for (int i = 1; i < argc; ++i) {
        arg += argv[i];
    }
    // std::cout << arg << std::endl;
    std::cout << parser.parse(arg) << std::endl;
}
