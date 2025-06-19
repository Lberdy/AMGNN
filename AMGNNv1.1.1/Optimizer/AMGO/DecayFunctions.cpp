#include<cmath>

class TimeBasedDecay{

public:
    double decayRate = 0.005;

    double calculateLR(double LR, int epoch){

        return LR/(1+decayRate*epoch);

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&decayRate), sizeof(double));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&decayRate), sizeof(double));
    }

};

class StepDecay{

public:
    double dropRate = 0.5;
    int dropEvery = 10;

    double calculateLR(double LR, int epoch){

        return LR*pow(dropRate, floor((double)epoch/dropEvery));

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&dropRate), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&dropEvery), sizeof(int));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&dropRate), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&dropEvery), sizeof(int));
    }

};

class ExponentialDecay{

public:
    double decayRate = 0.05;

    double calculateLR(double LR, int epoch){

        return LR*exp(-decayRate*epoch);

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&decayRate), sizeof(double));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&decayRate), sizeof(double));
    }

};

class PolynomialDecay{

public:
    double finalLR = 0.00001;
    double power = 2;

    double calculateLR(double LR, int epoch, int maxEpoch){

        return (LR - finalLR)*pow((1 - (double)epoch/maxEpoch), power) + finalLR;

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&finalLR), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&power), sizeof(double));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&finalLR), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&power), sizeof(double));
    }

};

class CosineAnnealingDecay{

public:
    double minLR = 0.00001;

    double calculateLR(double LR ,int epoch, int maxEpoch){

        return minLR + 0.5*(LR - minLR)*(1 + cos(M_PI*(double)epoch/maxEpoch));

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&minLR), sizeof(double));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&minLR), sizeof(double));
    }

};