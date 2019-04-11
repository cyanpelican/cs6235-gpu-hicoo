#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include "coo.hpp"
#include "common.hpp"


std::vector<double> split(const std::string *str, char delimiter) {
    std::vector<double> internal;
    std::stringstream ss(*str); // Turn the string into a stream.
    std::string tok;

    while(getline(ss, tok, delimiter)) {
        internal.push_back(stod(tok));
    }

    return internal;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: <program name> <sparse matrix file name>\n");
        exit(1);
    }

    char *matrixName = argv[1];
    CooTensorManager* Coo = new CooTensorManager();
    Coo->create(argv[1]);


    return 0;
}

