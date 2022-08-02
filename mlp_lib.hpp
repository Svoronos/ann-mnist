/* 
 * File:   mlp_lib.hpp
 * Author: Svoronos Leivadaros
 *
 * 
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

#include <cstdlib> 
#include <ctime>
#include <string>
#include <vector>
#include <cstring>
#include <ostream> 
#include <chrono>
#include <iostream>

#ifndef MLP_LIB_HPP
#define MLP_LIB_HPP
#define RANGE_RAND_LOW (-1)
#define RANGE_RAND_HIGH (1)
#define RESERVED_SPACE 256

double randomRange(double low, double high);

double getSigmoidValue(double x);
double getReLU(double x);
double getLeakyReLU(double x);

double getSigmoidDerivative(double x);
double getReLUDerivative(double x);
double getLeakyReLUDerivative(double x);

//SEMVER versioning system
uint32_t majorVersion = 1;
uint32_t minorVersion = 0;
uint32_t patchVersion = 0;

//Enumerate the type of a layer 

enum NeuronLayerType {
    input = 0, hidden, output
};

enum ActivationFunction {
    sigmoid = 0, ReLU, LeakyReLU
};


std::string lutLayerTypes[] = {"input", "hidden", "output"};
std::string lutActivationFunctions[] = {"Sigmoid", "Rectified Linear Unit", "Leaky Rectified Linear Unit"};

/*
 * Class representation for a set of weights
 */
class WeightSet {
    double *weightsArray; //an array of weights for a pair of layers
    int layer1Neurons; //the number of neurons in the left layer
    int layer2Neurons; //the number of neurons in the right layer
public:

    WeightSet(double* weightsArray, int layer1Neurons, int layer2Neurons) :
    weightsArray(weightsArray), layer1Neurons(layer1Neurons), layer2Neurons(layer2Neurons) {
    }

    /**
     * Initializer function for class WeightSet
     * @param low : the lowest possible real value that will be assigned to a weight
     * @param high : the highest possible real value that will be assigned to a weight
     * @param layer1Neurons : the number of neurons on the left layer
     * @param layer2Neurons : the number of neurons on the right layer
     */
    WeightSet(double low, double high, int layer1Neurons, int layer2Neurons) :
    layer1Neurons(layer1Neurons), layer2Neurons(layer2Neurons) {
        weightsArray = new double[layer1Neurons * layer2Neurons]; //allocate memory to store the set of weights in a flattened 2D array
        for (int j = 0; j < layer1Neurons; j++) { //each row is the weights connecting the j_th neuron of the left layer to all the neurons on the right layer
            for (int k = 0; k < layer2Neurons; k++) {
                weightsArray[j * layer2Neurons + k] = randomRange(low, high); //assign a random value in the range [low,high]
            }
        }
    }

    /**
     * Same initializer as above, but set all weights to value 0
     * @param layer1Neurons : the number of neurons on the left layer
     * @param layer2Neurons : the number of neurons on the right layer
     */
    WeightSet(int layer1Neurons, int layer2Neurons) :
    layer1Neurons(layer1Neurons), layer2Neurons(layer2Neurons) {
        weightsArray = new double[layer1Neurons * layer2Neurons];
        for (int j = 0; j < layer1Neurons; j++) {
            for (int k = 0; k < layer2Neurons; k++) {
                weightsArray[j * layer2Neurons + k] = 0.0;
            }
        }
    }

    int GetLayer1Neurons() const {
        return layer1Neurons;
    }

    void SetLayer1Neurons(int layer1Neurons) {
        this->layer1Neurons = layer1Neurons;
    }

    int GetLayer2Neurons() const {
        return layer2Neurons;
    }

    void SetLayer2Neurons(int layer2Neurons) {
        this->layer2Neurons = layer2Neurons;
    }

    double* GetWeightsArray() const {
        return weightsArray;
    }

    void SetWeightsArray(double* weightsArray) {
        this->weightsArray = weightsArray;
    }

    double getWeightAtIndex(int layer1Index, int layer2Index) {
        return this->weightsArray[layer1Index * layer2Neurons + layer2Index];
    }

    void setWeightAtIndex(int layer1Index, int layer2Index, double newWeight) {
        this->weightsArray[layer1Index * layer2Neurons + layer2Index] = newWeight;
    }

    void print() {
        std::cout << "Left layer comprised of " << layer1Neurons << " neurons." << std::endl;
        std::cout << "Right layer comprised of " << layer2Neurons << " neurons." << std::endl;
        for (int i = 0; i < 1; i++) {
            for (int j = 0; j < 8; j++) {
                std::cout << "L_neuron" << i << ":R_neuron" << j << " = " << weightsArray[i * layer2Neurons + j] << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

class BiasSet {
    double *biasArray;
    int layerNeurons;
public:

    BiasSet(double* biasArray, int layerNeurons) :
    biasArray(biasArray), layerNeurons(layerNeurons) {
    }

    BiasSet(int layerNeurons, double low, double high) :
    layerNeurons(layerNeurons) {
        biasArray = new double[layerNeurons];
        for (int i = 0; i < layerNeurons; i++) {
            biasArray[i] = randomRange(low, high);
        }
    }

    double* getBiasArray() const {
        return biasArray;
    }

    void setBiasArray(double* biasArray) {
        this->biasArray = biasArray;
    }

    int getLayerNeurons() const {
        return layerNeurons;
    }

    void setLayerNeurons(int layerNeurons) {
        this->layerNeurons = layerNeurons;
    }

    double getBiasAtIndex(int index) {
        return this->biasArray[index];
    }

    void setBiasAtIndex(int index, double newBias) {
        this->biasArray[index] = newBias;
    }

    void print() {
        std::cout << "Bias array for " << layerNeurons << " neurons." << std::endl;
        for (int i = 0; i < 8; i++) {
            std::cout << "Bias " << i << " = " << biasArray[i ] << std::endl;
        }
        std::cout << std::endl;
    }
};

class NeuronSet {
    double *neuronsOutputs;
    double *neuronDeltas;
    int numberOfNeurons;
    NeuronLayerType type;
public:

    NeuronSet(double* neuronsOutputs, int numberOfNeurons, NeuronLayerType type) :
    neuronsOutputs(neuronsOutputs), numberOfNeurons(numberOfNeurons), type(type) {
    }

    NeuronSet(int numberOfNeurons, NeuronLayerType type) :
    numberOfNeurons(numberOfNeurons), type(type) {
        neuronsOutputs = new double[numberOfNeurons];
        neuronDeltas = new double[numberOfNeurons];
    }

    double getOutputAtIndex(int i) {
        return neuronsOutputs[i];
    }

    void setOutputAtIndex(int i, double output) {
        this->neuronsOutputs[i] = output;
    }

    double getDeltaAtIndex(int i) {
        return neuronDeltas[i];
    }

    void setDeltaAtIndex(int i, double delta) {
        this->neuronDeltas[i] = delta;
    }

    double* GetNeuronsOutputs() const {
        return neuronsOutputs;
    }

    void SetNeuronsOutputs(double* neuronsOutputs) {
        this->neuronsOutputs = neuronsOutputs;
    }

    double* getNeuronDeltas() const {
        return neuronDeltas;
    }

    void setNeuronDeltas(double* neuronDeltas) {
        this->neuronDeltas = neuronDeltas;
    }

    int GetNumberOfNeurons() const {
        return numberOfNeurons;
    }

    void SetNumberOfNeurons(int numberOfNeurons) {
        this->numberOfNeurons = numberOfNeurons;
    }

    NeuronLayerType GetType() const {
        return type;
    }

    void SetType(NeuronLayerType type) {
        this->type = type;
    }

    void print() {
        std::cout << "Neuron set, number of neurons = " << numberOfNeurons << std::endl;
        std::cout << "Type of layer = " << lutLayerTypes[type] << std::endl;
        for (int i = 0; i < numberOfNeurons; i++) {
            std::cout << neuronsOutputs[i] << std::endl;
        }
    }

};

/*
 * Class definition for a Neural Network
 * It consists of 
 *          A set of weights (L-1 set of weights where L is the number of layers in the network)
 *          A set of biases (L-1 set of biases where L is the number of layers in the network)
 *          A set of layers of neurons (each layer is consisted of any number of neurons)
 *          Definition of an activation function (Sigmoid, ReLU etc.)
 *          Learning rate (not necessary to include as a class member?)
 */
class NeuralNetwork {
    std::vector<WeightSet> weightSets;
    std::vector<BiasSet> biasSets;
    std::vector<NeuronSet> layers;
    ActivationFunction activationFunction;
    int inputLayerSize;
    int outputLayerSize;
    double learningRate;
    int64_t totalNumberOfWeightsAndBiases;
    int64_t totalNumberOfWeights;
    int64_t totalNumberOfBiases;
public:

    /*
     * Class initializer, accepts as input 
     *      an array of integers denoting the architecture of the network, 
     *      the range of values to randomly generate and initialize the weights and biases
     *      the activation function to use
     *      the learning rate to use during training
     */
    NeuralNetwork(int *architectureLayout, int architectureSize, double lowRangeRandom, double highRangeRandom, ActivationFunction activationFunction, double learnRate) {
        inputLayerSize = architectureLayout[0];
        outputLayerSize = architectureLayout[architectureSize - 1];
        totalNumberOfWeights = 0;
        totalNumberOfBiases = 0;
        for (int i = 0; i < (architectureSize - 1); i++) {
            bool isOutput = i == (architectureSize - 2);
            WeightSet *weightSet = new WeightSet(lowRangeRandom, highRangeRandom, architectureLayout[i], architectureLayout[i + 1]);
            BiasSet *biasSet = new BiasSet(architectureLayout[i + 1], lowRangeRandom, highRangeRandom);
            weightSets.push_back(*weightSet);
            biasSets.push_back(*biasSet);
            totalNumberOfWeights += architectureLayout[i] * architectureLayout[i + 1];
            totalNumberOfBiases += architectureLayout[i + 1];
        }
        totalNumberOfWeightsAndBiases = totalNumberOfWeights + totalNumberOfBiases;
        NeuronLayerType type;

        for (int i = 0; i < architectureSize; i++) {
            if (i == 0) {
                type = NeuronLayerType::input;
            } else if (i == (architectureSize - 1)) {
                type = NeuronLayerType::output;
            } else {
                type = NeuronLayerType::hidden;
            }
            NeuronSet *layer = new NeuronSet(architectureLayout[i], type);
            layers.push_back(*layer);
        }
        this->activationFunction = activationFunction;
        this->learningRate = learnRate;
    }

    /*
     * Class initializer that parses a .bin binary model file that was created during training of the network
     * Usually called to parse a trained model to memory and utilize it to predict what handwritten digit is on viewed in an image
     */
    NeuralNetwork(std::string modelFileName) {
        char buffer[1000];
        std::ifstream modelFile(modelFileName, std::ios::in | std::ios::binary);

        //read the version number of the training software used to train the model
        modelFile.read(buffer, sizeof (uint32_t));
        uint32_t *majorVersionPtr = (uint32_t *) buffer;
        uint32_t majorVersionModelRead = *majorVersionPtr;
        modelFile.read(buffer, sizeof (uint32_t));
        uint32_t *minorVersionPtr = (uint32_t *) buffer;
        uint32_t minorVersionModelRead = *minorVersionPtr;
        modelFile.read(buffer, sizeof (uint32_t));
        uint32_t *patchVersionPtr = (uint32_t *) buffer;
        uint32_t patchVersionModelRead = *patchVersionPtr;
        std::cout << "Parsed model file " << modelFileName << " trained with software version " << majorVersionModelRead << "." << minorVersionModelRead << "." << patchVersionModelRead << "." << std::endl;

        if (modelIsFutureVersion(majorVersionModelRead, minorVersionModelRead, patchVersionModelRead)) {
            std::cout << "[CRITICAL WARNING] : Model's version (" << majorVersionModelRead << "." << minorVersionModelRead << "." << patchVersionModelRead  <<  ")  is higher than current software version ("  << majorVersion << "." << minorVersion << "." << patchVersion << "), erratic behaviour may occur!\n";
        }
        //read the number of layers present in the network
        modelFile.read(buffer, sizeof (size_t));
        size_t *ptrSizeT = (size_t *) buffer;
        size_t numberOfLayers = *ptrSizeT;
        int *architectureLayout = new int[numberOfLayers];
        int numberOfWeightSets = numberOfLayers - 1;
        int numberOfBiasSets = numberOfWeightSets;

        //read the enumerated type of activation function
        modelFile.read(buffer, sizeof (int));
        int *actFuncInt = (int *) buffer;
        ActivationFunction actFunc = static_cast<ActivationFunction> (*actFuncInt);

        //read the reserved space bytes
        int reservedSpaceSize = RESERVED_SPACE;
        modelFile.read(buffer, reservedSpaceSize);

        //read layer size for each layer
        for (int i = 0; i < numberOfLayers; i++) {
            modelFile.read(buffer, sizeof (int));
            int *layerSize = (int *) buffer;
            architectureLayout[i] = *layerSize;
        }


        //parse all weights in the file
        inputLayerSize = architectureLayout[0];
        outputLayerSize = architectureLayout[numberOfLayers - 1];
        totalNumberOfWeights = 0;
        totalNumberOfBiases = 0;
        for (int i = 0; i < numberOfWeightSets; i++) {
            WeightSet *weightSet = new WeightSet(architectureLayout[i], architectureLayout[i + 1]);
            for (int j = 0; j < architectureLayout[i]; j++) {
                for (int k = 0; k < architectureLayout[i + 1]; k++) {
                    modelFile.read(buffer, sizeof (double));
                    double *currentWeightValue = (double *) buffer;
                    weightSet->setWeightAtIndex(j, k, *currentWeightValue);
                }
            }
            weightSets.push_back(*weightSet);
            totalNumberOfWeights += architectureLayout[i] * architectureLayout[i + 1];
        }

        for (int i = 0; i < numberOfBiasSets; i++) {
            BiasSet *biasSet = new BiasSet(architectureLayout[i + 1], 0, 0);
            for (int j = 0; j < architectureLayout[i + 1]; j++) {
                modelFile.read(buffer, sizeof (double));
                double *currentBiasValue = (double *) buffer;
                biasSet->setBiasAtIndex(j, *currentBiasValue);
            }

            biasSets.push_back(*biasSet);
            totalNumberOfBiases += architectureLayout[i + 1];
        }
        totalNumberOfWeightsAndBiases = totalNumberOfWeights + totalNumberOfBiases;

        NeuronLayerType type;

        for (int i = 0; i < numberOfLayers; i++) {
            if (i == 0) {
                type = NeuronLayerType::input;
            } else if (i == (numberOfLayers - 1)) {
                type = NeuronLayerType::output;
            } else {
                type = NeuronLayerType::hidden;
            }
            NeuronSet *layer = new NeuronSet(architectureLayout[i], type);
            layers.push_back(*layer);
        }
        this->activationFunction = actFunc;
        this->learningRate = 0.01;
    }

    //Copy constructor for class NeuralNetwork

    NeuralNetwork(const NeuralNetwork& other) :
    weightSets(other.weightSets), biasSets(other.biasSets), layers(other.layers), activationFunction(other.activationFunction), inputLayerSize(other.inputLayerSize), outputLayerSize(other.outputLayerSize), learningRate(other.learningRate), totalNumberOfWeightsAndBiases(other.totalNumberOfWeightsAndBiases), totalNumberOfWeights(other.totalNumberOfWeights), totalNumberOfBiases(other.totalNumberOfBiases) {
    }

    /**
     * Function that is used to initialize the input layer of the network with the real-representation values of the MNIST images
     * Called for each image that we need to either train the network on or test it
     * @param values : pointer to double-representation pixels of the current image processed
     * @param size : number of values to initialize (should be 28*28 for the MNIST dataset)
     */
    void initializeInputLayer(double *values, int size) {
        if (size != inputLayerSize) {
            std::cout << "ERROR : when initializing input layer, number of input values must be equal to number of input layer neurons (" << inputLayerSize << ")\n";
        }

        NeuronSet inputLayer = layers.at(0);
        for (int i = 0; i < size; i++) {
            inputLayer.setOutputAtIndex(i, values[i]);
        }
    }

    /**
     * Function called when we want to train the network
     * @param epochs : how many epochs to run the training session for
     * @param doubleImagesTrain : double pointer to the training dataset (pixel values converted to double representation using mnistConvertToDouble function)
     * @param arrayLabelsTrain  : uint8_t pointer to labels of the training dataset ([0,9] digit labels)
     * @param doubleImagesTest  : double pointer to the testing dataset (pixel values converted to double representation using mnistConvertToDouble function)
     * @param arrayLabelsTest   : uint8_t pointer to labels of the testing dataset ([0,9] digit labels)
     * @param totalTrain        : how many images to use for the training dataset (max is 60000 in the original dataset)
     * @param totalTest         : how many images to use for the testing dataset  (max is 10000 in the original dataset)
     */
    void train(int epochs, double *doubleImagesTrain, uint8_t *arrayLabelsTrain, double *doubleImagesTest, uint8_t *arrayLabelsTest, int totalTrain, int totalTest) {
        std::chrono::steady_clock::time_point beginEpoch;
        std::chrono::steady_clock::time_point endEpoch;
        double maxTestAccuracy = 0;
        double meanSquareErrorTrain = 0;
        double meanSquareErrorTest = 0;
        int maxTestAccuracyEpoch = 0;

        int hitsTrain, hitsTest;
        //BEGIN TRAINING
        for (int epoch = 0; epoch < epochs; epoch++) {
            beginEpoch = std::chrono::steady_clock::now();
            hitsTrain = 0;
            hitsTest = 0;
            meanSquareErrorTest = 0;
            meanSquareErrorTrain = 0;
            //Firstly, pass through the training dataset
            for (int i = 0; i < totalTrain; i++) {
                int imagesIndex = i * (28 * 28);

                this->initializeInputLayer(&doubleImagesTrain[imagesIndex], (28 * 28));
                this->forwardPropagate();
                meanSquareErrorTrain += this->getMeanSquareError(arrayLabelsTrain[i + 8]);
                hitsTrain += this->isCorrectOutput(arrayLabelsTrain[i + 8]);
                this->backPropagate(arrayLabelsTrain[i + 8]);
                double percentageEpoch = ((double) i / (double) totalTrain) * 100;
                if (!(i % 100)) { //progress bar printing every 100 images
                    printf("%5.02f%% current epoch training...\r", percentageEpoch);
                }
            }
            meanSquareErrorTrain /= totalTrain;
            printf("                                                                                       \r");

            for (int i = 0; i < totalTest; i++) {
                int imagesIndex = i * (28 * 28);
                this->initializeInputLayer(&doubleImagesTest[imagesIndex], (28 * 28));
                this->forwardPropagate();
                meanSquareErrorTest += this->getMeanSquareError(arrayLabelsTest[i + 8]);
                hitsTest += this->isCorrectOutput(arrayLabelsTest[i + 8]);
                double percentageEpochTesting = ((double) i / (double) totalTest) * 100;

                //progress bar
                printf("%5.02f%% current epoch testing...\r", percentageEpochTesting);

            }
            meanSquareErrorTest /= totalTest;
            double accuracyPercentageTrain = 100 * ((double) hitsTrain / (double) totalTrain);
            double accuracyPercentageTest = 100 * ((double) hitsTest / (double) totalTest);

            //After training is completed on the current epoch, if the accuracy on the test data is higher than the
            //Highest accuracy observed thus far, write the model to the binary model file (always keep the model with the highest test accuracy)
            if (maxTestAccuracy < accuracyPercentageTest) {
                maxTestAccuracy = accuracyPercentageTest;
                maxTestAccuracyEpoch = epoch;
                this->writeModel("model.bin");
            }
            endEpoch = std::chrono::steady_clock::now();
            uint64_t nanosEpoch = std::chrono::duration_cast<std::chrono::nanoseconds>(endEpoch - beginEpoch).count();
            double secondsEpoch = nanosEpoch / 1000000000.;
            printf("( %6.1f sec ) Epoch %04d Accuracy_Train %6.2f%% Accuracy_Test %6.2f%% Max_Acc_Test %6.2f%% @epoch %3d MSE_Train %6.4f MSE_Test %6.4f\n", secondsEpoch, epoch, accuracyPercentageTrain, accuracyPercentageTest, maxTestAccuracy, maxTestAccuracyEpoch, meanSquareErrorTrain, meanSquareErrorTest);
        }
    }

    /**
     * Function used to forward propagate through the network, assumes the input layer has been previously initialized
     * with {@initializeInputLayer()} function.
     */
    void forwardPropagate() {
        //We only cycle through the layers and biases using the weight index currently processing
        //We assume layers and biases were correctly initialized when the network was created 
        //TODO : add checks to see that bias sizes and neuronset size are the correct values?
        int weightSize = weightSets.size();
        double newOutput;
        for (int wIndex = 0; wIndex < weightSize; wIndex++) {
            WeightSet weights = weightSets.at(wIndex);
            BiasSet biases = biasSets.at(wIndex);
            NeuronSet leftLayer = layers.at(wIndex);
            NeuronSet rightLayer = layers.at(wIndex + 1);

            //Use an array of sums to store the products of each weight and neuron output sets
            //This is used to ensure the access on each weight set is sequential to increase cache hit rate
            double *sums = new double[rightLayer.GetNumberOfNeurons()];
            memset(sums, 0, rightLayer.GetNumberOfNeurons() * sizeof (double));
            for (int i = 0; i < leftLayer.GetNumberOfNeurons(); i++) {
                double sum = 0;
                for (int j = 0; j < rightLayer.GetNumberOfNeurons(); j++) {
                    sums[j] += weights.getWeightAtIndex(i, j) * leftLayer.getOutputAtIndex(i);
                }
            }

            //For each right layer node, take its computed sum of products and add the bias to it, 
            //then use the selected activation function on the final sum result
            for (int i = 0; i < rightLayer.GetNumberOfNeurons(); i++) {
                sums[i] += biases.getBiasAtIndex(i);
                if (activationFunction == ActivationFunction::sigmoid) {
                    newOutput = getSigmoidValue(sums[i]);
                } else if (activationFunction == ActivationFunction::ReLU) {
                    newOutput = getReLU(sums[i]);
                } else if (activationFunction == ActivationFunction::LeakyReLU) {
                    newOutput = getLeakyReLU(sums[i]);
                }
                rightLayer.setOutputAtIndex(i, newOutput);
            }
            delete[] sums;
        }
    }

    /**
     * Back propagation function, called after calling the forward propagation function
     * Called to calculate the error deltas for each neuron and change the weights accordingly
     * @param trueLabel
     */
    void backPropagate(uint8_t trueLabel) {
        int weightSize = weightSets.size();

        //Output layer back propagate, different method than hidden layers backprop
        for (int wIndex = weightSize - 1; wIndex >= (weightSize - 1); wIndex--) {
            NeuronSet leftLayer = layers.at(wIndex);
            NeuronSet rightLayer = layers.at(wIndex + 1);
            WeightSet weightCurrent = weightSets.at(wIndex);
            BiasSet biasCurrent = biasSets.at(wIndex);
            double *deltaCurrent = new double[weightCurrent.GetLayer2Neurons()];
            WeightSet weights = weightSets.at(wIndex);

            for (int i = 0; i < weightCurrent.GetLayer2Neurons(); i++) {
                double labelExpected = (i == trueLabel) ? 1. : 0.;
                deltaCurrent[i] = rightLayer.getOutputAtIndex(i) - labelExpected;
                rightLayer.setDeltaAtIndex(i, deltaCurrent[i]);
            }

            for (int i = 0; i < weightCurrent.GetLayer1Neurons(); i++) {
                for (int j = 0; j < weightCurrent.GetLayer2Neurons(); j++) {
                    double oldWeight = weights.getWeightAtIndex(i, j);
                    double newWeight = oldWeight + (-learningRate * deltaCurrent[j] * leftLayer.getOutputAtIndex(i));
                    weights.setWeightAtIndex(i, j, newWeight);
                }

            }
            for (int i = 0; i < weightCurrent.GetLayer2Neurons(); i++) {
                double oldBias = biasCurrent.getBiasAtIndex(i);
                double newBias = oldBias + (-learningRate * deltaCurrent[i]);
                biasCurrent.setBiasAtIndex(i, newBias);
            }

            delete[] deltaCurrent;
        }

        //Hidden layers backpropagation, delta computation
        for (int wIndex = weightSize - 2; wIndex >= 0; wIndex--) {
            NeuronSet leftLayer = layers.at(wIndex);
            NeuronSet rightLayer = layers.at(wIndex + 1);
            NeuronSet rightRightLayer = layers.at(wIndex + 2);
            WeightSet weightCurrent = weightSets.at(wIndex);
            WeightSet weightNext = weightSets.at(wIndex + 1);
            BiasSet biasCurrent = biasSets.at(wIndex);
            double *sums = new double[weightCurrent.GetLayer2Neurons()];
            memset(sums, 0, weightCurrent.GetLayer2Neurons() * sizeof (double));

            //apply the delta rule for the next set of weights
            for (int i = 0; i < rightRightLayer.GetNumberOfNeurons(); i++) {
                for (int j = 0; j < weightCurrent.GetLayer2Neurons(); j++) {
                    sums[j] += weightNext.getWeightAtIndex(j, i) * rightRightLayer.getDeltaAtIndex(i);
                }
            }

            //apply the delta rule computed in previous step to get current layer's error
            for (int i = 0; i < weightCurrent.GetLayer2Neurons(); i++) {
                double deltaNow;
                if (activationFunction == ActivationFunction::sigmoid) {
                    deltaNow = sums[i] * getSigmoidDerivative(rightLayer.getOutputAtIndex(i));
                } else if (activationFunction == ActivationFunction::ReLU) {
                    deltaNow = sums[i] * getReLUDerivative(rightLayer.getOutputAtIndex(i));
                } else if (activationFunction == ActivationFunction::LeakyReLU) {
                    deltaNow = sums[i] * getLeakyReLUDerivative(rightLayer.getOutputAtIndex(i));
                }
                rightLayer.setDeltaAtIndex(i, deltaNow);
            }

            //Change the weights according to the learning rate and the delta error of each neuron computed in previous step
            for (int i = 0; i < weightCurrent.GetLayer1Neurons(); i++) {
                for (int j = 0; j < weightCurrent.GetLayer2Neurons(); j++) {
                    double oldWeight = weightCurrent.getWeightAtIndex(i, j);
                    double newWeight = oldWeight + (-learningRate * rightLayer.getDeltaAtIndex(j) * leftLayer.getOutputAtIndex(i));
                    weightCurrent.setWeightAtIndex(i, j, newWeight);
                }

            }

            //Correct the biases
            for (int i = 0; i < weightCurrent.GetLayer2Neurons(); i++) {
                double oldBias = biasCurrent.getBiasAtIndex(i);
                double newBias = oldBias + (-learningRate * rightLayer.getDeltaAtIndex(i));
                biasCurrent.setBiasAtIndex(i, newBias);
            }
            delete[] sums;
        }
    }

    /**
     * Function used to check if the neuron with the max value at the output layer of a forward-propagated network 
     * is the same as the current true label. Assumes that the forwardPropagate function has been called beforehand
     * @param trueLabel : a value between 0 and 9 denoting the true value of the currently trained/tested image
     * @return  : true if the value is predicted correctly, false otherwise
     */
    bool isCorrectOutput(uint8_t trueLabel) {
        NeuronSet outputLayer = layers.at(layers.size() - 1);
        double maxValue = outputLayer.getOutputAtIndex(0);
        uint8_t maxIndex = 0;
        for (int i = 0; i < outputLayer.GetNumberOfNeurons(); i++) {
            if (maxValue < outputLayer.getOutputAtIndex(i)) {
                maxValue = outputLayer.getOutputAtIndex(i);
                maxIndex = i;
            }
        }
        return trueLabel == maxIndex;
    }

    /**
     * Function used to predict an image's displayed digit. 
     * Assumes the image is of size 28*28 and has been converted to double representation beforehand.
     * @param values : array of double-precision floating numbers representing pixel values of image
     * @param size : number of pixels to initialize the input layer 
     * @param confidencePercentage : the confidence percentage of the max-value class at the output layer
     * @return : the class that the image is predicted to belong to (digits 0 to 9 in the MNIST dataset case)
     */
    uint8_t predict(double *values, int size, double *confidencePercentage) {
        initializeInputLayer(values, size);
        forwardPropagate();

        NeuronSet outputLayer = layers.at(layers.size() - 1);
        double maxValue = outputLayer.getOutputAtIndex(0);
        uint8_t maxIndex = 0;
        for (int i = 0; i < outputLayer.GetNumberOfNeurons(); i++) {
            if (maxValue < outputLayer.getOutputAtIndex(i)) {
                maxValue = outputLayer.getOutputAtIndex(i);
                maxIndex = i;
            }
        }
        *confidencePercentage = maxValue * 100;
        return maxIndex;
    }

    /**
     * Function used to calculate the mean square error of a forward-propagated network in correspondance to a true value
     * @param trueLabel : a value between 0 and 9 denoting the true value of the currently trained/tested image
     * @return  : real value, the result of the mean square error function
     */
    double getMeanSquareError(uint8_t trueLabel) {
        NeuronSet outputLayer = layers.at(layers.size() - 1);
        double sum = 0;
        for (int i = 0; i < outputLayer.GetNumberOfNeurons(); i++) {
            if (i == trueLabel) {
                sum += (outputLayer.getOutputAtIndex(i) - 1.) * (outputLayer.getOutputAtIndex(i) - 1.);
            } else {
                sum += (outputLayer.getOutputAtIndex(i) - 0.) * (outputLayer.getOutputAtIndex(i) - 0.);
            }
        }
        return sum / 2.;
    }

    /**
     * Function that is used to write the model's weights and biases in a binary file format
     * This binary file will then be able to be parsed by the NeuralNetwork constructor with a string argument
     * and written in memory so it can be used to predict which digit is shown in a 28*28 image (like the MNIST dataset sizes)
     * 
     * This constructor uses a binary file to parse and initialize itself according to the values written to it
     * see @NeuralNetwork(string modelFileName) function for more info 
     * 
     * Argument 'writeFileName' : the name/path destination of the binary file to write the neural network on
     */
    void writeModel(std::string writeFileName) {
        std::fstream modelFile(writeFileName, std::ios::out | std::ios::binary);
        char *ptrChar;

        //write current library version with which the current model was trained with
        ptrChar = (char *) &majorVersion;
        modelFile.write(ptrChar, sizeof (uint32_t));
        ptrChar = (char *) &minorVersion;
        modelFile.write(ptrChar, sizeof (uint32_t));
        ptrChar = (char *) &patchVersion;
        modelFile.write(ptrChar, sizeof (uint32_t));


        //write the number of layers the neural network is comprised of
        size_t numberOfLayers = this->layers.size();
        ptrChar = (char *) &numberOfLayers;
        modelFile.write(ptrChar, sizeof (size_t));

        //write the enumarated type of activation function used in the network
        int typeOfActivationFunction = this->activationFunction;
        ptrChar = (char *) &typeOfActivationFunction;
        modelFile.write(ptrChar, sizeof (int));

        //write reserved space 
        int reservedSpaceSize = RESERVED_SPACE;
        char reservedSpace[reservedSpaceSize];
        memset(reservedSpace, 0, reservedSpaceSize);
        modelFile.write(reservedSpace, reservedSpaceSize);

        //write the size of each layer in the binary file
        for (int i = 0; i < this->layers.size(); i++) {
            NeuronSet layer = this->layers.at(i);
            int numberOfNeurons = layer.GetNumberOfNeurons();
            ptrChar = (char *) &numberOfNeurons;
            modelFile.write(ptrChar, sizeof (int));
        }


        //write all the weight values in double format for each set of weights
        int numberOfWeightSets = this->layers.size() - 1;
        for (int i = 0; i < numberOfWeightSets; i++) {
            WeightSet weightSet = this->weightSets.at(i);
            for (int j = 0; j < weightSet.GetLayer1Neurons(); j++) {
                for (int k = 0; k < weightSet.GetLayer2Neurons(); k++) {
                    double curValue = weightSet.getWeightAtIndex(j, k);
                    ptrChar = (char *) &curValue;
                    modelFile.write(ptrChar, sizeof (double));
                }
            }
        }

        //write all the bias values in double format for each set of biases
        int numberOfBiasSets = numberOfWeightSets; //we have as many bias sets as we have weight sets
        for (int i = 0; i < numberOfBiasSets; i++) {
            BiasSet biasSet = this->biasSets.at(i);
            for (int j = 0; j < biasSet.getLayerNeurons(); j++) {
                double curBias = biasSet.getBiasAtIndex(j);
                ptrChar = (char *) &curBias;
                modelFile.write(ptrChar, sizeof (double));
            }
        }
    }

    /**
     * Check if a parsed model's version is higher than the current software parsing it
     * @param modelMajor : SemVer major version number
     * @param modelMinor : SemVer minor version number
     * @param modelPatch : SemVer patch version number
     * @return : true if the model's version is higher, else false
     */
    bool modelIsFutureVersion(uint32_t modelMajor, uint32_t modelMinor, uint32_t modelPatch) {
        if (modelMajor < majorVersion) {
            return false;
        } else if (modelMajor > majorVersion) {
            return true;
        } else {
            if (modelMinor < minorVersion) {
                return false;
            } else if (modelMinor > minorVersion) {
                return true;
            } else {
                if (modelPatch < patchVersion) {
                    return false;
                } else if (modelPatch > patchVersion) {
                    return true;
                }
            }
        }
        
        return false; //all checks failed, thus the version of the model is exactly EQUAL to the current software's
    }

    std::string getActivationFunction() {
        return lutActivationFunctions[this->activationFunction];
    }

    int getTotalNumberOfBiases() const {
        return totalNumberOfBiases;
    }

    void setTotalNumberOfBiases(int totalNumberOfBiases) {
        this->totalNumberOfBiases = totalNumberOfBiases;
    }

    int getTotalNumberOfWeights() const {
        return totalNumberOfWeights;
    }

    void setTotalNumberOfWeights(int totalNumberOfWeights) {
        this->totalNumberOfWeights = totalNumberOfWeights;
    }

    int getTotalNumberOfWeightsAndBiases() const {
        return totalNumberOfWeightsAndBiases;
    }

    void setTotalNumberOfWeightsAndBiases(int totalNumberOfWeightsAndBiases) {
        this->totalNumberOfWeightsAndBiases = totalNumberOfWeightsAndBiases;
    }

};

double randomRange(double low, double high) {
    double rand0To1 = (double) rand() / (double) RAND_MAX;
    double diff = high - low;
    double offset = (rand0To1 * diff) + low;
    return offset;
}

double getSigmoidValue(double x) {
    return 1. / (1 + exp(-x));
}

double getReLU(double x) {
    return std::max(0., x);
}

double getLeakyReLU(double x) {
    if (x >= 0) {
        return x;
    } else {
        return 0.1 * x;
    }
}

double getSigmoidDerivative(double x) {
    return x * (1 - x);
}

double getReLUDerivative(double x) {
    return std::max(0., x);
}

double getLeakyReLUDerivative(double x) {
    if (x > 0) {
        return 1.;
    } else if (x < 0) {
        return 0.1;
    } else {
        return 0.5;
    }
}

#endif /* MLP_LIB_HPP */


