#include"../../Methods/LBFGS.cpp"
#include<mutex>

std::mutex adamMutex;

class Adam{

    int weightNumber = 0;
    int weightOptimized = 0;
    int currentStep = 1;
    std::vector<double> Moment1;
    std::vector<double> Moment2;

public:
    double B1 = 0.9;
    double B2 = 0.999;
    double epsilon = 0.00000001;
    double cachedB1 = B1;
    double cachedB2 = B2;

    Adam(const NN& nn){

        for(const HiddenLayer& layer:nn.HiddenLayers){
            weightNumber += layer.InWeights.size();
            weightNumber += layer.biases.size();
        }

        weightNumber += nn.outputLayer.InWeights.size();
        weightNumber += nn.outputLayer.biases.size();

        Moment1 = Matrix::zeros(weightNumber);
        Moment2 = Matrix::zeros(weightNumber);

    }

    Adam(){}

    double optimizeWeight(double weight, double gradient, double LR, int num){

        double M = B1*Moment1[num] + (1 - B1)*gradient;
        double V = B2*Moment2[num] + (1 - B2)*(gradient*gradient);

        double Mbc = M/(1 - cachedB1);
        double Vbc = V/(1 - cachedB2);

        double stepSize = LR*(Mbc/(sqrt(Vbc) + epsilon));
        double newWeight = weight - stepSize;

        Moment1[num] = M;
        Moment2[num] = V;

        {
            std::lock_guard<std::mutex> lock(adamMutex);
            weightOptimized++;
            if(weightOptimized == weightNumber){
                currentStep++;
                weightOptimized = 0;
                cachedB1 *= B1;
                cachedB2 *= B2;
            }
        }

        return newWeight;

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&weightNumber), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&B1), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&B2), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&epsilon), sizeof(double));

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&weightNumber), sizeof(int));
        Moment1 = Matrix::zeros(weightNumber);
        Moment2 = Matrix::zeros(weightNumber);
        fileIn.read(reinterpret_cast<char*>(&B1), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&B2), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&epsilon), sizeof(double));

    }

};