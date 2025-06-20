#include"StochasticGradientDescent.cpp"
#include<algorithm>

class MiniBatchGradientDescent{

public:
    int BatchSize = 30;
    size_t parallelBatches = 3;

    template<class OPT>
    void optimize(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, OPT& opt){

        ThreadPooling batchThreadPool(parallelBatches);

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!opt.isEnd(nn, epoch, InputData, OutputData)){

            std::vector<std::vector<double>> InputCopy = InputData;
            std::vector<std::vector<double>> OutputCopy = OutputData;
            std::vector<std::vector<std::vector<std::vector<double>>>> Batches;

            while(InputCopy.size() > 0){

                std::vector<std::vector<double>> InputBatch;
                std::vector<std::vector<double>> OutputBatch;
                std::vector<std::vector<std::vector<double>>> Batch;

                int size = 0;
                while(size < BatchSize && InputCopy.size() > 0){
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
                    [this,&nn,BatchPtr,epoch,&opt](){
                        opt.optimize(nn,(*BatchPtr)[0],(*BatchPtr)[1],epoch);
                    }
                );
            }

            batchThreadPool.wait_for_all_tasks();

            epoch++;

        }

    }

};