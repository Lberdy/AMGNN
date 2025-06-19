#include"DecayFunctions.cpp"

enum DecayFunctions{

    TIME_BASED_DECAY,
    STEP_DECAY,
    EXPONENTIAL_DECAY,
    POLYNOMIAL_DECAY,
    COSINE_ANNEALING_DECAY

};

class AMGO{

public:
    bool DecayFunction = true;
    DecayFunctions DecayFunctionName = EXPONENTIAL_DECAY;

    TimeBasedDecay TBD;
    StepDecay SD;
    ExponentialDecay ED;
    PolynomialDecay PD;
    CosineAnnealingDecay CAD;

    double optimizeWeight(double weight, double gradient, double LR, int epoch, int maxEpoch){

        double learningRate;

        if(DecayFunction){

            switch(DecayFunctionName){
                case TIME_BASED_DECAY : learningRate = TBD.calculateLR(LR, epoch); break;
                case STEP_DECAY : learningRate = SD.calculateLR(LR, epoch); break;
                case EXPONENTIAL_DECAY : learningRate = ED.calculateLR(LR, epoch); break;
                case POLYNOMIAL_DECAY : learningRate = PD.calculateLR(LR, epoch, maxEpoch); break;
                case COSINE_ANNEALING_DECAY : learningRate = CAD.calculateLR(LR, epoch, maxEpoch); break;
            }

        }else{
            learningRate = LR;
        }

        double stepSize = learningRate*gradient;
        double newWeight = weight - stepSize;

        return newWeight;

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&DecayFunction), sizeof(bool));
        int DFN = static_cast<int>(DecayFunctionName);
        fileOut.write(reinterpret_cast<char*>(&DFN), sizeof(int));

        TBD.save(fileOut);
        SD.save(fileOut);
        ED.save(fileOut);
        PD.save(fileOut);
        CAD.save(fileOut);

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&DecayFunction), sizeof(bool));
        int DFN;
        fileIn.read(reinterpret_cast<char*>(&DFN), sizeof(int));
        DecayFunctionName = static_cast<DecayFunctions>(DFN);

        TBD.load(fileIn);
        SD.load(fileIn);
        ED.load(fileIn);
        PD.load(fileIn);
        CAD.load(fileIn);

    }

};