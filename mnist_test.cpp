/**
 *
 MIT License

Copyright (c) 2022 Svoronos Leivadaros

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */

#include <iostream>
#include <math.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <getopt.h>
#include <vector>
#include <chrono>

#include "mlp_lib.hpp"
#include "utils.hpp"

int main(int argc, char * argv[]) {
    srand(static_cast<double> (std::time(nullptr))); //TODO : better random function!
    std::cout.precision(12);

    int opt;
    int64_t totalTest = 10000;
    char buffer[200];
    std::string modelFileNameParse = "model.bin";
    while ((opt = getopt(argc, argv, "m:")) != -1) {
        switch (opt) {
            case 'm':
                strcpy(buffer, optarg);
                modelFileNameParse = std::string(buffer);
                break;
            default:
            case 'h':
                printf("Usage: %s [-m]\n", argv[0]);
                printf("\tArg -m STRING : name of multi-layer perceptron mode file\n");
                printf("\tArg -h : display this help message\n");
                return 1;
        }
    }


    std::cout << "Reading testing dataset..." << std::endl;
    uint8_t *arrayImagesTest = parseMnistFile("test-images.idx");
    uint8_t *arrayLabelsTest = parseMnistFile("test-labels.idx");
    std::cout << "Parsed training and testing dataset onto memory..." << std::endl;


    std::cout << "Converting images to double representation in memory..." << std::endl;
    int offsetImages = 16;
    double *doubleImagesTest = new double[totalTest * MNIST_HEIGHT * MNIST_WIDTH];

    mnistConvertToDouble(doubleImagesTest, &arrayImagesTest[offsetImages], totalTest);
    std::cout << "Converted images to double representation!" << std::endl;


    NeuralNetwork annParsed(modelFileNameParse);
    int hitsTest = 0;
    double confidencePercentage;
    for (int i = 0; i < totalTest; i++) {
        int imagesIndex = i * (28 * 28);
        uint8_t trueClass = arrayLabelsTest[i + 8];
        uint8_t predictedClass = annParsed.predict(&doubleImagesTest[imagesIndex], MNIST_HEIGHT * MNIST_WIDTH, &confidencePercentage);
        if(predictedClass == trueClass){
            hitsTest++;
        }
        double percentageEpochTesting = ((double) i / (double) totalTest) * 100;
        printf("%5.02f%% current epoch testing...\r", percentageEpochTesting);

    }
    double accuracyPercentageTest = 100 * ((double) hitsTest / (double) totalTest);
    std::cout << "Accuracy " << accuracyPercentageTest << "% on parsed binary model file." << std::endl;
    return 1;
}

