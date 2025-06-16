#include"../../Differentiation.cpp"

std::mutex GradientMutex;
std::condition_variable cvar;

class LBFGS{

    int weights;
    std::vector<double> MemoryGradients;
    std::vector<double> MemoryWeights;
    std::vector<std::vector<std::vector<double>>> Memory_S_Y;
    std::vector<double> Memory_p;
    Differentiation differentiation;
    LossFunction lossFunction;

public:
    int m = 20;

    LBFGS(Differentiation diff, LossFunction lf){
        differentiation = diff;
        lossFunction = lf;
    }

    LBFGS(){}

private:

    std::vector<double> getCurrentWeights(NN& nn){

        std::vector<double> weights;

        for(HiddenLayer& layer:nn.HiddenLayers){
            for(double value:layer.InWeights){
                weights.push_back(value);
            }

            for(double value:layer.biases){
                weights.push_back(value);
            }
        }

        for(double value:nn.outputLayer.InWeights){
            weights.push_back(value);
        }

        for(double value:nn.outputLayer.biases){
            weights.push_back(value);
        }

        return weights;

    }

    std::vector<double> calculateGradients(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations){

        std::vector<double> gradients(weights);
        int num = 0;

        ThreadPooling threadpool(paralelOperations);

        for(int i = 0; i < nn.HiddenLayers.size(); i++){
            HiddenLayer& layer = nn.HiddenLayers[i];
            for(int j = 0; j < layer.InWeights.size(); j++){
                threadpool.enqueue(
                    [this, &nn, &InputData, &OutputData, &gradients, i, j, num](){
                        double gradient = differentiation.derivate(nn, i, j, true, false, lossFunction, InputData, OutputData);
                        gradients[num] = gradient;
                    }
                );
                num++;
            }

            for(int j = 0; j < layer.biases.size(); j++){
                threadpool.enqueue(
                    [this, &nn, &InputData, &OutputData, &gradients, i, j, num](){
                        double gradient = differentiation.derivate(nn, i, j, false, false, lossFunction, InputData, OutputData);
                        gradients[num] = gradient;
                    }
                );
                num++;
            }
        }


        for(int i = 0; i < nn.outputLayer.InWeights.size(); i++){
            threadpool.enqueue(
                [this, &nn, &InputData, &OutputData, &gradients, i, num](){
                    double gradient = differentiation.derivate(nn, 0, i, true, true, lossFunction, InputData, OutputData);
                    gradients[num] = gradient;
                }
            );
            num++;
        }

        for(int i = 0; i < nn.outputLayer.biases.size(); i++){
            threadpool.enqueue(
                [this, &nn, &InputData, &OutputData, &gradients, i, num](){
                    double gradient = differentiation.derivate(nn, 0, i, false, true, lossFunction, InputData, OutputData);
                    gradients[num] = gradient;
                }
            );
            num++;
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

    void initialization(NN& nn, double learning_rate, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations){

        MemoryWeights = getCurrentWeights(nn);
        weights = MemoryWeights.size();
        MemoryGradients = calculateGradients(nn, InputData, OutputData, paralelOperations);

        updateNN(nn, MemoryGradients, learning_rate);

        std::vector<double> currentWeights = getCurrentWeights(nn);
        std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations);

        updateMemory(currentWeights, currentGradients);

    }

    bool End(NN& nn, int iteration, int iterations, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

        if(iteration == iterations){
            return true;
        }

        double loss = lossFunction.calculateLoss(nn, InputData, OutputData);
        std::cout<<"loss : "<<loss<<std::endl;
        if(loss <= lossEpsilon){
            return true;
        }

        return false;

    }

    void optimizing(NN& nn, double learning_rate, int iterations, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations){

        int iteration = 1;

        while(!End(nn, iteration, iterations, lossEpsilon, InputData, OutputData)){

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
            std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations);

            updateMemory(currentWeights, currentGradients);

            iteration++;

        }

    }

public:
    void optimize(NN& nn, int iterations, double learning_rate, double lossEpsilon, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations){

        differentiation.InterpolationPolynomialOrder = ORDER_6;

        initialization(nn, learning_rate, InputData, OutputData, paralelOperations);

        optimizing(nn, learning_rate, iterations, lossEpsilon, InputData, OutputData, paralelOperations);

    }

};