#include"LossFunction.cpp"
#include"Optimizer/ThreadPooling.cpp"

enum Order{
    ORDER_2,
    ORDER_4,
    ORDER_6
};

class Differentiation{

public:
    double h = 0.005;
    Order InterpolationPolynomialOrder = ORDER_4;

private:
    template<class XNN, class INPUT>
    double CalculateFunction(XNN& nn, double x, double hc, int index, LossFunction& func, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        nn.changeValue(index, x+hc*h);
        double result = func.calculateLoss(nn, InputData, OutputData);
        nn.restoreValue();

        return result;

    }

    template<class XNN, class INPUT>
    double derivateOrder2(XNN& nn, double x, int index, LossFunction& func, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){

        double f1 = CalculateFunction(nn, x, 1, index, func, InputData, OutputData);
        double f2 = CalculateFunction(nn, x, -1, index, func, InputData, OutputData);

        double result = f1 - f2;
        result = result/(2*h);

        return result;

    }

    template<class XNN, class INPUT>
    double derivateOrder4(XNN& nn, double x, int index, LossFunction& func, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){

        double f1 = CalculateFunction(nn, x, -2, index, func, InputData, OutputData);
        double f2 = CalculateFunction(nn, x, -1, index, func, InputData, OutputData);
        double f3 = CalculateFunction(nn, x, 1, index, func, InputData, OutputData);
        double f4 = CalculateFunction(nn, x, 2, index, func, InputData, OutputData);

        double result = f1 - 8*f2 + 8*f3 - f4;
        result = result/(12*h);

        return result;

    }

    template<class XNN, class INPUT>
    double derivateOrder6(XNN& nn, double x, int index, LossFunction& func, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){

        double f1 = CalculateFunction(nn, x, -3, index, func, InputData, OutputData);
        double f2 = CalculateFunction(nn, x, -2, index, func, InputData, OutputData);
        double f3 = CalculateFunction(nn, x, -1, index, func, InputData, OutputData);
        double f4 = CalculateFunction(nn, x, 1, index, func, InputData, OutputData);
        double f5 = CalculateFunction(nn, x, 2, index, func, InputData, OutputData);
        double f6 = CalculateFunction(nn, x, 3, index, func, InputData, OutputData);

        double result = -f1 + 9*f2 - 45*f3 + 45*f4 - 9*f5 + f6;
        result = result/(60*h);

        return result;

    }

public:
    template<class XNN, class INPUT>
    double derivate(XNN& nn, int index, LossFunction& func, std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){

        auto nnCopy = nn.deepCopy(nn);

        double x = *nnCopy.FlattenWeights[index];

        double result;

        switch(InterpolationPolynomialOrder){
            case ORDER_2 : result = derivateOrder2(nnCopy, x, index, func, InputData, OutputData); break;
            case ORDER_4 : result = derivateOrder4(nnCopy, x, index, func, InputData, OutputData); break;
            case ORDER_6 : result = derivateOrder6(nnCopy, x, index, func, InputData, OutputData); break;
        }

        return result;

    }

    void save(std::ofstream& fileOut){

        fileOut.write(reinterpret_cast<char*>(&h), sizeof(double));

        int IPO = static_cast<int>(InterpolationPolynomialOrder);
        fileOut.write(reinterpret_cast<char*>(&IPO), sizeof(int));

    }

    void load(std::ifstream& fileIn){

        fileIn.read(reinterpret_cast<char*>(&h), sizeof(double));

        int IPO;
        fileIn.read(reinterpret_cast<char*>(&IPO), sizeof(int));
        InterpolationPolynomialOrder = static_cast<Order>(IPO);

    }

};