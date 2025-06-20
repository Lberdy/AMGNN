#include"GradientDescent.cpp"
#include<algorithm>

class StochasticGradientDescent{

public:
    template<class OPT>
    void optimize(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, OPT& opt){

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

        while(!opt.isEnd(nn, epoch, InputData, OutputData)){

            std::vector<std::vector<std::vector<double>>> DataSetCopy = DataSet;

            shuffle(DataSetCopy.begin(), DataSetCopy.end(), engine);
            
            while(DataSetCopy.size() > 0){

                std::uniform_int_distribution<int> destribution(0,DataSetCopy.size()-1);
                int index = destribution(engine);

                std::vector<std::vector<double>> input = {DataSetCopy[index][0]};
                std::vector<std::vector<double>> output = {DataSetCopy[index][1]};

                opt.optimize(nn, input, output, epoch);

                DataSetCopy.erase(DataSetCopy.begin() + index);
            }

            epoch++;

        }

    }

};