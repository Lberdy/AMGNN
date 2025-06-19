#include"../../Differentiation.cpp"

std::mutex GradientMutex;
std::condition_variable cvar;

class LBFGS{

    int weights;
    std::vector<double> MemoryGradients;
    std::vector<double> MemoryWeights;
    std::vector<std::vector<std::vector<double>>> Memory_S_Y;
    std::vector<double> Memory_p;

public:
    int m = 20;


private:

    std::vector<double> getCurrentWeights(NN& nn){

        std::vector<double> weights;

        for(int i = 0; i < nn.weightsFlatten.size(); i++){
            weights.push_back(*nn.weightsFlatten[i]);
        }

        return weights;

    }

    std::vector<double> calculateGradients(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation& diff, LossFunction& lossF){

        std::vector<double> gradients(weights);

        ThreadPooling threadpool(paralelOperations);

        for(int i = 0; i < weights; i++){
            threadpool.enqueue(
                [this, &nn, &InputData, &OutputData, &gradients, i, &diff, &lossF]{
                    double gradient = diff.derivate(nn, i, lossF, InputData, OutputData);
                    gradients[i] = gradient;
                }
            );
        }

        threadpool.wait_for_all_tasks();

        return gradients;

    }

    void updateNN(NN& nn, std::vector<double>& P, double learning_rate){

        int num = 0;

        for(HiddenLayer& layer:nn.HiddenLayers){
            for(int i = 0; i < layer.InWeights.size(); i++){
                layer.InWeights[i] -= P[num]*learning_rate;
                num++; 
            }

            for(int i = 0; i < layer.biases.size(); i++){
                layer.biases[i] -= P[num]*learning_rate;
                num++;
            }
        }

        for(int i = 0; i < nn.outputLayer.InWeights.size(); i++){
            nn.outputLayer.InWeights[i] -= P[num]*learning_rate;
            num++;
        }

        for(int i = 0; i < nn.outputLayer.biases.size(); i++){
            nn.outputLayer.biases[i] -= P[num]*learning_rate;
            num++;
        }

    }

    void updateMemory(std::vector<double>& currentWeights, std::vector<double>& currentGradients){

        std::vector<double> S;
        std::vector<double> Y;

        for(int i = 0; i < currentWeights.size(); i++){
            S.push_back(currentWeights[i] - MemoryWeights[i]);
        }

        for(int i = 0; i < currentGradients.size(); i++){
            Y.push_back(currentGradients[i] - MemoryGradients[i]);
        }

        double p = 1/Matrix::fastDot(Y, S);

        std::vector<std::vector<double>> pair;
        pair.push_back(S);
        pair.push_back(Y);

        if(Memory_S_Y.size() == m){
            Memory_S_Y.erase(Memory_S_Y.begin());
            Memory_p.erase(Memory_p.begin());
        }

        Memory_S_Y.push_back(pair);
        Memory_p.push_back(p);

        MemoryWeights = currentWeights;
        MemoryGradients = currentGradients;

    }

    void initialization(NN& nn, double learning_rate, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation& diff, LossFunction& lossF){

        MemoryWeights = getCurrentWeights(nn);
        weights = MemoryWeights.size();
        MemoryGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, diff, lossF);

        updateNN(nn, MemoryGradients, learning_rate);

        std::vector<double> currentWeights = getCurrentWeights(nn);
        std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, diff, lossF);

        updateMemory(currentWeights, currentGradients);

    }

    bool End(NN& nn, int iteration, int iterations, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, LossFunction lossF){

        if(iteration == iterations){
            return true;
        }

        double loss = lossF.calculateLoss(nn, InputData, OutputData);
        std::cout<<"loss : "<<loss<<std::endl;
        if(loss <= lossEpsilon){
            return true;
        }

        return false;

    }

    void optimizing(NN& nn, double learning_rate, int iterations, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation diff, LossFunction lossF){

        int iteration = 1;

        while(!End(nn, iteration, iterations, lossEpsilon, InputData, OutputData, lossF)){

            std::vector<double> q = MemoryGradients;
            std::vector<double> alphas;

            for(int i = Memory_S_Y.size()-1; i >= 0; i--){
                std::vector<double>& S = Memory_S_Y[i][0];
                std::vector<double>& Y = Memory_S_Y[i][1];

                double alpha = Memory_p[i]*Matrix::fastDot(S,q);
                alphas.push_back(alpha);

                for(int j = 0; j < q.size(); j++){
                    q[j] -= alpha*Y[j];
                }

            }

            double v = Matrix::fastDot(Memory_S_Y.back()[0], Memory_S_Y.back()[1])/Matrix::fastDot(Memory_S_Y.back()[1], Memory_S_Y.back()[1]);
            std::vector<double> r;
            for(double value:q){
                r.push_back(value*v);
            }

            for(int i = 0; i < Memory_S_Y.size(); i++){
                std::vector<double>& S = Memory_S_Y[i][0];
                std::vector<double>& Y = Memory_S_Y[i][1];

                double beta = Memory_p[i]*Matrix::fastDot(Y,r);

                for(int j = 0; j < r.size(); j++){
                    r[j] += S[j]*(alphas[Memory_S_Y.size()-i-1] - beta);
                }
            }

            updateNN(nn, r, learning_rate);

            std::vector<double> currentWeights = getCurrentWeights(nn);
            std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, diff, lossF);

            updateMemory(currentWeights, currentGradients);

            iteration++;

        }

    }

public:
    void optimize(NN& nn, int iterations, double learning_rate, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation diff, LossFunction lossF){

        diff.InterpolationPolynomialOrder = ORDER_6;

        initialization(nn, learning_rate, InputData, OutputData, paralelOperations, diff, lossF);

        optimizing(nn, learning_rate, iterations, lossEpsilon, InputData, OutputData, paralelOperations, diff, lossF);

    }

};