#include"../../Differentiation.cpp"

std::mutex GradientMutex;
std::condition_variable cvar;

class LBFGS{

    std::vector<double> MemoryGradients;
    std::vector<double> MemoryWeights;
    std::vector<std::vector<std::vector<double>>> Memory_S_Y;
    std::vector<double> Memory_p;

public:
    int m = 20;

    LBFGS(){}

private:

    template<class XNN>
    std::vector<double> getCurrentWeights(XNN& nn){

        std::vector<double> weights;

        for(int i = 0; i < nn.FlattenWeights.size(); i++){
            weights.push_back(*nn.FlattenWeights[i]);
        }

        return weights;

    }

    template<class XNN, class INPUT>
    std::vector<double> calculateGradients(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation differentiation, LossFunction lossFunction){

        std::vector<double> gradients(nn.FlattenWeights.size());

        ThreadPooling threadpool(paralelOperations);

        for(int i = 0; i < nn.FlattenWeights.size(); i++){
            threadpool.enqueue(
                [this,&nn,&InputData,&OutputData,i,&gradients,&differentiation,&lossFunction](){
                    double gradient = differentiation.derivate(nn, i, lossFunction, InputData, OutputData);
                    gradients[i] = gradient;
                }
            );
        }

        threadpool.wait_for_all_tasks();

        return gradients;

    }

    template<class XNN>
    void updateNN(XNN& nn, std::vector<double>& P, double learning_rate){

        for(int i = 0; i < nn.FlattenWeights.size(); i++){
            *nn.FlattenWeights[i] -= P[i]*learning_rate;
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

    template<class XNN, class INPUT>
    void initialization(XNN& nn, double learning_rate, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation differentiation, LossFunction lossFunction){

        MemoryWeights = getCurrentWeights(nn);
        MemoryGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, differentiation, lossFunction);

        updateNN(nn, MemoryGradients, learning_rate);

        std::vector<double> currentWeights = getCurrentWeights(nn);
        std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, differentiation, lossFunction);

        updateMemory(currentWeights, currentGradients);

    }

    template<class XNN, class INPUT>
    bool End(XNN& nn, int iteration, int iterations, double lossEpsilon, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, LossFunction lossFunction){

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

    template<class XNN, class INPUT>
    void optimizing(XNN& nn, double learning_rate, int iterations, double lossEpsilon, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation differentiation, LossFunction lossFunction){

        int iteration = 1;

        while(!End(nn, iteration, iterations, lossEpsilon, InputData, OutputData, lossFunction)){

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
            std::vector<double> currentGradients = calculateGradients(nn, InputData, OutputData, paralelOperations, differentiation, lossFunction);

            updateMemory(currentWeights, currentGradients);

            iteration++;

        }

    }

public:
    template<class XNN, class INPUT>
    void optimize(XNN& nn, int iterations, double learning_rate, double lossEpsilon, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, size_t paralelOperations, Differentiation differentiation, LossFunction lossFunction){

        differentiation.InterpolationPolynomialOrder = ORDER_6;

        initialization(nn, learning_rate, InputData, OutputData, paralelOperations, differentiation, lossFunction);

        optimizing(nn, learning_rate, iterations, lossEpsilon, InputData, OutputData, paralelOperations, differentiation, lossFunction);

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&m), sizeof(int));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&m), sizeof(int));
    }

};