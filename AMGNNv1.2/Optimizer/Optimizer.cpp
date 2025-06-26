#include"Methods/MiniBatchGradientDescent.cpp"

enum Optimizers{
    AMGNNO,
    ADAM,
    LBFGSO
};

std::mutex nnMutex;

class Optimizer{

public:
    Optimizers OptimizerName;
    double learningRate = 0.01;
    double lossEpsilon = 0.01;
    int epoches;

    //Optmizers
    Adam adam;
    AMGO amgo;

    Differentiation differentiation;
    LossFunction lossFunction;

    size_t parallelOperations = 6;

    //Methods
    LBFGS lbfgs;
    GradientDescent GD;
    StochasticGradientDescent SGD;
    MiniBatchGradientDescent MBGD;
    
    template<class XNN>
    Optimizer(XNN& nn, Optimizers opt, int maxEpoches, Loss_Function LFN) : adam(nn), lossFunction(LFN){

        OptimizerName = opt;
        epoches = maxEpoches;

    }

    Optimizer(){}

public:
    template<class XNN, class INPUT>
    bool isEnd(XNN& nn, int epoch, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){

        if(epoch == epoches){
            return true;
        }
        
        double loss = lossFunction.calculateLoss(nn, InputData, OutputData);
        std::cout << "loss : "<<loss<<std::endl;
        if(loss <= lossEpsilon){
            return true;
        }

        return false;

    }

    template<class XNN, class INPUT>
    void optimize(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, int epoch){

        ThreadPooling threadpool(parallelOperations);

        for(int i = 0; i < nn.FlattenWeights.size(); i++){
            threadpool.enqueue(
                [this,&nn,&InputData,&OutputData,epoch,i](){
                    double gradient = differentiation.derivate(nn, i, lossFunction, InputData, OutputData);
                    {
                        std::lock_guard<std::mutex> lock(nnMutex);
                        double weight = *nn.FlattenWeights[i];
                        switch(OptimizerName){
                            case AMGNNO : *nn.FlattenWeights[i] = amgo.optimizeWeight(weight, gradient, learningRate, epoch, epoches); break;
                            case ADAM : *nn.FlattenWeights[i] = adam.optimizeWeight(weight, gradient, learningRate, i); break;
                        }
                    }
                }
            );
        }

        threadpool.wait_for_all_tasks();

    }

    template<class XNN, class INPUT>
    void gradientDescent(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        GD.optimize(nn, InputData, OutputData, *this);
    }

    template<class XNN, class INPUT>
    void stochasticGradientDescent(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        SGD.optimize(nn, InputData, OutputData, *this);
    }

    template<class XNN, class INPUT>
    void LBFGS_Optimizing(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        lbfgs.optimize(nn, epoches, learningRate, lossEpsilon, InputData, OutputData, parallelOperations, differentiation, lossFunction);
    }

    template<class XNN, class INPUT>
    void miniBatchGradientDescent(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        MBGD.optimize(nn, InputData, OutputData, *this);
    }

    void save(std::ofstream& fileOut){

        int OPT = static_cast<int>(OptimizerName);
        fileOut.write(reinterpret_cast<char*>(&OPT), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&learningRate), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&lossEpsilon), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&epoches), sizeof(int));

        lbfgs.save(fileOut);
        MBGD.save(fileOut);

        adam.save(fileOut);
        amgo.save(fileOut);
        differentiation.save(fileOut);
        lossFunction.save(fileOut);

    }

    void load(std::ifstream& fileIn){

        int OPT;
        fileIn.read(reinterpret_cast<char*>(&OPT), sizeof(int));
        OptimizerName = static_cast<Optimizers>(OPT);
        fileIn.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&lossEpsilon), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&epoches), sizeof(int));

        lbfgs.load(fileIn);
        MBGD.load(fileIn);

        adam.load(fileIn);
        amgo.load(fileIn);
        differentiation.load(fileIn);
        lossFunction.load(fileIn);

    }

};