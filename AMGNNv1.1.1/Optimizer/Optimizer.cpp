#include"ADAM/Adam.cpp"
#include"AMGO/AMGO.cpp"
#include<algorithm>

enum Optimizers{
    AMGNNO,
    ADAM,
    LBFGSO
};

std::mutex nnMutex;

class Optimizer{

public:
    int batches = 468;
    Optimizers OptimizerName;
    double learningRate = 0.01;
    double lossEpsilon = 0.01;
    int epoches;
    Adam adam;
    AMGO amgo;
    Differentiation differentiation;
    LossFunction lossFunction;
    size_t parallelOperations = 6;
    size_t parallelBatches = 3;
    LBFGS lbfgs;

    Optimizer(NN& nn, Optimizers opt, int maxEpoches, Loss_Function LFN) : adam(nn), lossFunction(LFN), lbfgs(differentiation, lossFunction){

        OptimizerName = opt;
        epoches = maxEpoches;

    }

    Optimizer(){}

private:
    bool isEnd(NN& nn, int epoch, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

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

    void optimize(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, int epoch){

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

public:
    void gradientDescent(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

        int epoch = 0;

        while(!isEnd(nn, epoch, InputData, OutputData)){

            optimize(nn, InputData, OutputData, epoch);

            epoch++;

        }

    }

    void stochasticGradientDescent(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

        std::vector<std::vector<std::vector<double>>> DataSet;
        for(int i = 0; i < InputData.size(); i++){
            std::vector<std::vector<double>> sample;
            sample.push_back(InputData[i]);
            sample.push_back(OutputData[i]);
            DataSet.push_back(sample);
        }

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!isEnd(nn, epoch, InputData, OutputData)){

            std::vector<std::vector<std::vector<double>>> DataSetCopy = DataSet;

            shuffle(DataSetCopy.begin(), DataSetCopy.end(), engine);
            
            while(DataSetCopy.size() > 0){

                std::uniform_int_distribution<int> destribution(0,DataSetCopy.size()-1);
                int index = destribution(engine);

                std::vector<std::vector<double>> input = {DataSetCopy[index][0]};
                std::vector<std::vector<double>> output = {DataSetCopy[index][1]};

                optimize(nn, input, output, epoch);

                DataSetCopy.erase(DataSetCopy.begin() + index);
            }

            epoch++;

        }

    }

    void LBFGS_Optimizing(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

        lbfgs.optimize(nn, epoches, learningRate, lossEpsilon, InputData, OutputData, parallelOperations);

    }

    void miniBatchGradientDescent(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData){

        ThreadPooling batchThreadPool(parallelBatches);

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!isEnd(nn, epoch, InputData, OutputData)){

            std::vector<std::vector<double>> InputCopy = InputData;
            std::vector<std::vector<double>> OutputCopy = OutputData;
            std::vector<std::vector<std::vector<std::vector<double>>>> Batches;

            while(InputCopy.size() > 0){

                std::vector<std::vector<double>> InputBatch;
                std::vector<std::vector<double>> OutputBatch;
                std::vector<std::vector<std::vector<double>>> Batch;

                int size = 0;
                while(size < batches && InputCopy.size() > 0){
                    std::uniform_int_distribution<int> distribution(0,InputCopy.size()-1);
                    int index = distribution(engine);
                    InputBatch.push_back(InputCopy[index]);
                    OutputBatch.push_back(OutputCopy[index]);

                    InputCopy.erase(InputCopy.begin() + index);
                    OutputCopy.erase(OutputCopy.begin() + index);
                    size++;
                }

                Batch.push_back(InputBatch);
                Batch.push_back(OutputBatch);
                Batches.push_back(Batch);
            }

            for(std::vector<std::vector<std::vector<double>>>& Batch:Batches){
                std::vector<std::vector<std::vector<double>>>* BatchPtr = &Batch;
                batchThreadPool.enqueue(
                    [this,&nn,BatchPtr,epoch](){
                        optimize(nn,(*BatchPtr)[0],(*BatchPtr)[1],epoch);
                    }
                );
            }

            batchThreadPool.wait_for_all_tasks();

            epoch++;

        }

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&batches), sizeof(int));
        int OPT = static_cast<int>(OptimizerName);
        fileOut.write(reinterpret_cast<char*>(&OPT), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&learningRate), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&lossEpsilon), sizeof(double));
        fileOut.write(reinterpret_cast<char*>(&epoches), sizeof(int));

        adam.save(fileOut);
        amgo.save(fileOut);
        differentiation.save(fileOut);
        lossFunction.save(fileOut);

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&batches), sizeof(int));
        int OPT;
        fileIn.read(reinterpret_cast<char*>(&OPT), sizeof(int));
        OptimizerName = static_cast<Optimizers>(OPT);
        fileIn.read(reinterpret_cast<char*>(&learningRate), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&lossEpsilon), sizeof(double));
        fileIn.read(reinterpret_cast<char*>(&epoches), sizeof(int));

        adam.load(fileIn);
        amgo.load(fileIn);
        differentiation.load(fileIn);
        lossFunction.load(fileIn);

    }

};