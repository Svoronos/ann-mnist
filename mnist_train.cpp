/**
 *  MIT License

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
#include <unistd.h>

#include "mlp_lib.hpp"
#include "utils.hpp"

#define DEFAULT_ARCHITECTURE_SIZE 3

#define MNIST_HEIGHT 28
#define MNIST_WIDTH 28

int main(int argc, char * argv[]) {
    srand(static_cast<double> (std::time(nullptr))); //@TODO : better random function!
    int opt;
    double learn_rate = 0.01;
    int epochs = 10;
    ActivationFunction activationFunction = ActivationFunction::sigmoid;
    int64_t totalTrain = 60000;
    int64_t totalTest = 10000;
    int *architectureLayout;
    //default architecture layout
    int architectureSize = DEFAULT_ARCHITECTURE_SIZE;
    architectureLayout = new int[DEFAULT_ARCHITECTURE_SIZE];
    architectureLayout[0] = 28 * 28; //input layer neurons (application-specific input layer size
    architectureLayout[1] = 32; //hidden layer neurons
    architectureLayout[2] = 10; //output layer neurons (application-specific, depends on classes we want to divide out neural network into)
    std::string delimiter = ",";
    std::vector<std::string> layersString;
    std::vector<std::string> rangeRandomString;
    int activationFunctionInt;
    char buffer[200];
    std::string arch;
    double lowRange = -0.05;
    double highRange = 0.05;

    //Parse arguments
    while ((opt = getopt(argc, argv, "a:l:e:f:r:h")) != -1) {
        switch (opt) {
            case 'a':
                strcpy(buffer, optarg);
                arch = std::string(buffer);
                layersString = parseDelimitedArgument(arch, delimiter);

                delete[] architectureLayout;
                architectureSize = layersString.size() + 2; //allocate architecture layout + 2 layers for input output
                architectureLayout = new int[architectureSize];
                architectureLayout[0] = (28 * 28);
                architectureLayout[architectureSize - 1] = 10;
                for (int i = 0; i < layersString.size(); i++) {
                    architectureLayout[i + 1] = std::stoi(layersString.at(i));
                }
                break;
            case 'l':
                learn_rate = atof(optarg);
                break;
            case 'e':
                epochs = atoi(optarg);
                break;
            case 'f':
                activationFunctionInt = atoi(optarg);
                if (activationFunctionInt < 0 || activationFunctionInt > 2) {
                    std::cout << "Error, wrong activation function argument, input 0 for sigmoid, 1 for ReLU, 2 for Leaky ReLU.\n";
                    return -1;
                }
                if (activationFunctionInt == 0) {
                    activationFunction = ActivationFunction::sigmoid;
                } else if (activationFunctionInt == 1) {
                    activationFunction = ActivationFunction::ReLU;
                } else if (activationFunctionInt == 2) {
                    activationFunction = ActivationFunction::LeakyReLU;
                }
                break;

            case 'r':
                strcpy(buffer, optarg);
                arch = std::string(buffer);
                rangeRandomString = parseDelimitedArgument(arch, ",");
                if (rangeRandomString.size() != 2) {
                    std::cout << "Accepted format for defining randomization range for weights is \n-0.5,0.5\n (a pair of real numbers separated by a ,)" << std::endl;
                }

                lowRange = std::stof(rangeRandomString.at(0));
                highRange = std::stof(rangeRandomString.at(1));
                break;
            default:
            case 'h':
                printf("Usage: %s [-a -l -e -f -r]\n", argv[0]);
                printf("\tArg -a STRING : architecture of neural network, example syntax : \n"
                        "\t\t%s -a 64,32,16\n"
                        "\t\tfor a neural network with 3 hidden layers,\n"
                        "\t\tfirst hidden layer with 64 neurons, second hidden layer with 32, third hidden layer with 16\n", argv[0]);
                printf("\tArg -l FLOAT: learning rate (e.g. 0.01)\n");
                printf("\tArg -e INTEGER: number of epochs to run training for\n");
                printf("\tArg -f INTEGER: activation function to use, 0 for Sigmoid, 1 for ReLU, 2 for Leaky ReLU\n");
                printf("\tArg -r comma-separate string: range of random values to initialize neural network weights and biases with\n"
                        "\t\te.g. '-2,2' to initialize weights and biases between -2 and 2\n");
                printf("\nExample : begin training with a neural network with architecture 784 > 128 > 128 > 10 a learning rate of 0.02 and randomized initial weights in the range [-0.2,0.2]\n"
                        "%s -a 128,128 -l 0.02 -r -0.2,0.2\n", argv[0]);

                return 1;
        }
    }

    std::cout << "ANN Mnist training application, software ver. " << majorVersion << "." << minorVersion << "." << patchVersion << "." << std::endl;
    NeuralNetwork ann(architectureLayout, architectureSize, lowRange, highRange, activationFunction, learn_rate);

    std::cout << "Reading training and testing dataset..." << std::endl;
    //Read images and labels of training and testing datasets onto memory
    uint8_t *arrayImagesTrain = parseMnistFile("train-images.idx");
    uint8_t *arrayLabelsTrain = parseMnistFile("train-labels.idx");

    uint8_t *arrayImagesTest = parseMnistFile("test-images.idx");
    uint8_t *arrayLabelsTest = parseMnistFile("test-labels.idx");
    std::cout << "Parsed training and testing dataset onto memory!" << std::endl;

    //Convert the training and testing images to double representation
    //to speed up initialization of input layer when training the neural network
    std::cout << "Converting images to double representation in memory..." << std::endl;
    int offsetImages = 16;
    double *doubleImagesTrain = new double[totalTrain * MNIST_HEIGHT * MNIST_WIDTH];
    double *doubleImagesTest = new double[totalTest * MNIST_HEIGHT * MNIST_WIDTH];

    mnistConvertToDouble(doubleImagesTrain, &arrayImagesTrain[offsetImages], totalTrain);
    mnistConvertToDouble(doubleImagesTest, &arrayImagesTest[offsetImages], totalTest);
    std::cout << "Converted images to double representation!" << std::endl;

    std::cout << "Neural network architecture : " << architectureLayout[0];
    for (int i = 1; i < architectureSize; i++) {
        std::cout << ">>" << architectureLayout[i];
    }
    std::cout << "." << std::endl;
    std::cout << "Running training of neural network for " << epochs << " epochs with learning rate of " << learn_rate <<
            " and activation function " << ann.getActivationFunction() << ". Range of randomized weights is [" << lowRange << "," << highRange << "]." << std::endl;

    std::cout << "Network number of weights and biases " << ann.getTotalNumberOfWeightsAndBiases() << std::endl;

    //begin training of the network
    ann.train(epochs, doubleImagesTrain, arrayLabelsTrain, doubleImagesTest, arrayLabelsTest, totalTrain, totalTest);

    return 0;
}
