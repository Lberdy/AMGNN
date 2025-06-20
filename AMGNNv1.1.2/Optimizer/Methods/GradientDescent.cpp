#include"../Optimizers/AMGO/AMGO.cpp"

class GradientDescent{

public:
    template<class OPT>
    void optimize(NN& nn, std::vector<std::vector<double>>& InputData, std::vector<std::vector<double>>& OutputData, OPT& opt){

        int epoch = 0;

        while(!opt.isEnd(nn, epoch, InputData, OutputData)){

            opt.optimize(nn, InputData, OutputData, epoch);

            epoch++;

        }

    }

};