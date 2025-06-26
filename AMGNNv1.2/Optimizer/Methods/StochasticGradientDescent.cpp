#include"GradientDescent.cpp"
#include<algorithm>

template<class INPUT>
struct Sample{
    INPUT sampleInput;
    std::vector<double> sampleOutput;
};

class StochasticGradientDescent{

public:
    template<class OPT, class XNN, class INPUT>
    void optimize(XNN& nn, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData, OPT& opt){

        std::vector<Sample<INPUT>> DataSet;
        for(int i = 0; i < InputData.size(); i++){
            Sample<INPUT> sample;
            sample.sampleInput = InputData[i];
            sample.sampleOutput = OutputData[i];
            DataSet.push_back(sample);
        }

        std::random_device rd;
        std::default_random_engine engine(rd());

        int epoch = 0;

        while(!opt.isEnd(nn, epoch, InputData, OutputData)){

            std::vector<Sample<INPUT>> DataSetCopy = DataSet;

            shuffle(DataSetCopy.begin(), DataSetCopy.end(), engine);
            
            while(DataSetCopy.size() > 0){

                std::uniform_int_distribution<int> destribution(0,DataSetCopy.size()-1);
                int index = destribution(engine);

                std::vector<INPUT> input = {DataSetCopy[index].sampleInput};
                std::vector<std::vector<double>> output = {DataSetCopy[index].sampleOutput};

                opt.optimize(nn, input, output, epoch);

                DataSetCopy.erase(DataSetCopy.begin() + index);
            }

            epoch++;

        }

    }

};