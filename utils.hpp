/* 
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

#ifndef UTILS_HPP
#define UTILS_HPP
#include <unistd.h>
#define DEFAULT_ARCHITECTURE_SIZE 3
#define MNIST_HEIGHT 28
#define MNIST_WIDTH 28

std::vector<std::string> parseDelimitedArgument(std::string archArgument, std::string delimiter);
uint8_t *parseMnistFile(std::string fileName);
void mnistConvertToDouble(double *doubleRepresentationArray, uint8_t *unsigned8IntArray, int size);
bool fileExists(std::string fileName);

/**
 * Function used to parse the architecture or the range of random values to initialize weights of a new network 
 * E.g. convert '32,32' which denotes to create a neural network of architecture 784 >> 32 >> 32 >> 10
 * to a vector of strings
 * @param archArgument : string argument passed when calling the main function, a 'delimiter' separated set of numerical values
 * @param delimiter : the delimiter with which the values are to be split, usually a comma (',') 
 * @return : vector of strings, e.g. the input '64,32,16' will be converted into a vector of strings, length 3
 *              where first string is '64', second string is '32' and third string is '16'
 */
std::vector<std::string> parseDelimitedArgument(std::string archArgument, std::string delimiter){
    size_t pos = 0;
    std::string token;
    std::vector<std::string> layers;
    while ((pos = archArgument.find(delimiter)) != std::string::npos) {
        token = archArgument.substr(0, pos);
        layers.push_back(token);
        archArgument.erase(0, pos + delimiter.length());
    }

    layers.push_back(archArgument);

    return layers;
}

/**
 * Function used to read and parse a mnist .idx file
 * The file is parsed and stored in an array of 8-bit unsigned integers
 * @param fileName : name\path of the .idx file to parse
 * @return : pointer to first position of the uint8_t array that was used to store the contents of the .idx file
 */

uint8_t *parseMnistFile(std::string fileName){
    std::ifstream mnistFile(fileName, std::ios::in | std::ios::binary);

    mnistFile.seekg(0, std::ios::end); // seek to the end of the file 
    int sizeBytesImages = mnistFile.tellg();
    mnistFile.seekg(0, std::ios::beg); // seek to the start of the file 

    uint8_t *arrayInImages = new uint8_t[sizeBytesImages];

    mnistFile.read((char *) arrayInImages, sizeBytesImages);
    return arrayInImages;
}

/**
 * Function used to convert the 8-bit unsigned int pixel values of the MNIST images to a real number representation
 * The numbers are rescaled in the range [0,1]
 * @param doubleRepresentationArray : pointer to an array of double values, the destination array where the converted pixel values are to be stored
 * @param unsigned8IntArray : source array of pixel values from the mnist dataset
 * @param size : total number of images (note : NOT bytes) to cycle through and convert
 */
void mnistConvertToDouble(double *doubleRepresentationArray, uint8_t *unsigned8IntArray, int size){
    int uintArrayIndex = 0;
    for (int imageIndex = 0; imageIndex < size; imageIndex++) {
        int imageOffset = imageIndex * MNIST_HEIGHT * MNIST_WIDTH;
        for (int rowI = 0; rowI < MNIST_HEIGHT; rowI++) {
            int rowOffset = rowI * MNIST_WIDTH;
            for (int colI = 0; colI < MNIST_WIDTH; colI++) {
                doubleRepresentationArray[imageOffset + rowOffset + colI] = ((double) unsigned8IntArray[uintArrayIndex++]) / 255.;
            }
        }
    }
}


/**
 * Check if a file exists
 * @param fileName : name of file to check if it exists
 * @return : true if file exists, false otherwise
 */
bool fileExists(std::string fileName) {
    return ( access(fileName.c_str(), F_OK) != -1);
}

#endif /* UTILS_HPP */



