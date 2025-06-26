#include"StochasticGradientDescent.cpp"
#include<algorithm>

template<class INPUT>
struct BATCH {
    std::vector<INPUT> InputData;
    std::vector<std::vector<double>> OutputData;
};

class MiniBatchGradientDescent{

public:
    int BatchSize = 30;
    size_t parallelBatches = 3;

    template<class OPT, class XNN, class INPUT>
    void optimize(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, OPT& opt){

        ThreadPooling batchThreadPool(parallelBatches);

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!opt.isEnd(nn, epoch, InputData, OutputData)){

            std::vector<INPUT> InputCopy = InputData;
            std::vector<std::vector<double>> OutputCopy = OutputData;
            std::vector<BATCH<INPUT>> Batches;

            while(InputCopy.size() > 0){

                BATCH<INPUT> Batch;

                int size = 0;
                while(size < BatchSize && InputCopy.size() > 0){
                    std::uniform_int_distribution<int> distribution(0,InputCopy.size()-1);
                    int index = distribution(engine);

                    Batch.InputData.push_back(InputCopy[index]);
                    Batch.OutputData.push_back(OutputCopy[index]);

                    InputCopy.erase(InputCopy.begin() + index);
                    OutputCopy.erase(OutputCopy.begin() + index);
                    size++;
                }

                Batches.push_back(Batch);
            }

            for(BATCH<INPUT>& Batch:Batches){
                BATCH<INPUT>* BatchPtr = &Batch;
                batchThreadPool.enqueue(
                    [this,&nn,BatchPtr,epoch,&opt](){
                        opt.optimize(nn, (*BatchPtr).InputData, (*BatchPtr).OutputData, epoch);
                    }
                );
            }

            batchThreadPool.wait_for_all_tasks();

            epoch++;

        }

    }

    void save(std::ofstream& fileOut){
        fileOut.write(reinterpret_cast<char*>(&BatchSize), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&parallelBatches), sizeof(parallelBatches));
    }

    void load(std::ifstream& fileIn){
        fileIn.read(reinterpret_cast<char*>(&BatchSize), sizeof(int));
        fileIn.read(reinterpret_cast<char*>(&parallelBatches), sizeof(parallelBatches));
    }

};