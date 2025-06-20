#include"NN/NN.cpp"

enum Loss_Function{
    MSE,
    MAE,
    BINARY_CROSS_ENTROPY,
    CROSS_ENTROPY
};

class LossFunction{

    Loss_Function LossFunctionName;

public:
    LossFunction(Loss_Function LFN){

        LossFunctionName = LFN;

    }

    LossFunction(){}

private:
    double mse(double yi, double pi){
        return pow(yi - pi, 2);
    }

    double mae(double yi, double pi){
        double mean = yi - pi;
        if(mean >= 0){
            return mean;
        }else{
            return -mean;
        }
    }

    double binaryCrossEntropy(double yi, double pi){
        return -yi*log(pi) - (1-yi)*log(1-pi);
    }

    double crossEntropy(double yi, double pi){
        return -yi*log(pi);
    }

public:
    double calculateLoss(NN& nn, std::vector<std::vector<double>>& InputDataSet, std::vector<std::vector<double>>& OutputDataSet){

        double lossSum = 0;
        for(int i = 0; i < InputDataSet.size(); i++){
            std::vector<double> input = InputDataSet[i];
            std::vector<double> output = OutputDataSet[i];
            std::vector<double> predicted = nn.predict(input);
            double sum = 0;
            for(int j = 0; j < output.size(); j++){
                switch(LossFunctionName){
                    case MSE : sum += mse(output[j], predicted[j]); break;
                    case MAE : sum += mae(output[j], predicted[j]); break;
                    case BINARY_CROSS_ENTROPY : sum += binaryCrossEntropy(output[j], predicted[j]); break;
                    case CROSS_ENTROPY : sum += crossEntropy(output[j], predicted[j]); break;
                }
            }
            lossSum += sum;
        }

        return lossSum/InputDataSet.size();

    }

    void save(std::ofstream& fileOut){

        int LFN = static_cast<int>(LossFunctionName);
        fileOut.write(reinterpret_cast<char*>(&LFN), sizeof(int));

    }

    void load(std::ifstream& fileIn){

        int LFN;
        fileIn.read(reinterpret_cast<char*>(&LFN), sizeof(int));
        LossFunctionName = static_cast<Loss_Function>(LFN);

    }

};