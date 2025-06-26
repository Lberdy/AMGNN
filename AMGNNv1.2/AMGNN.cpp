#include"Optimizer/Optimizer.cpp"

enum LearningMethod{
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT,
    LBFGS_Method
};

class AMGNN{

public:
    NN nn;
    CNN cnn;
    Optimizer optimizer;
    TaskType taskType;
    LearningMethod learningMethod;

    static AMGNN NeuralNetwork(ActivationFunction AFN, TaskType TT, int InputNeurons, std::vector<int> HiddenNeurons, int OutputNeurons, Loss_Function LFN, Optimizers opt, int Epoches, LearningMethod LRM){

        AMGNN amgnn(AFN, TT, InputNeurons, HiddenNeurons, OutputNeurons, LFN, opt, Epoches, LRM);

        return amgnn;

    }

    static AMGNN ConvolutionalNeuralNetwork(std::vector<std::vector<int>> kernelsInfos, std::vector<std::vector<int>> poolingsInfos, std::vector<int> padding, ActivationFunction CAFN, int channels, bool GAP, std::vector<int> InputSize, ActivationFunction DAFN, std::vector<int> HiddenNeurons, int outputNeurons, TaskType taskType, Loss_Function LFN, Optimizers opt, int Epoches, LearningMethod LRM){

        AMGNN amgnn(kernelsInfos, poolingsInfos, padding, CAFN, channels, GAP, InputSize, DAFN, HiddenNeurons, outputNeurons, taskType, LFN, opt, Epoches, LRM);
        return amgnn;

    }

    static cv::Mat readImage(std::string path, cv::ImreadModes mode){

        cv::Mat img = cv::imread(path, mode);
        if(img.empty()){
            std::cout<<"the image is empty, go play away kid\n";
            exit(1);
        }

        cv::Mat imgDouble;
        img.convertTo(imgDouble, CV_64F, 1.0/255.0);

        return imgDouble;

    }

private:
    AMGNN(ActivationFunction AFN, TaskType TT, int InputNeurons, std::vector<int> HiddenNeurons, int OutputNeurons, Loss_Function LFN, Optimizers opt, int Epoches, LearningMethod LRM)
    : nn(AFN, TT, InputNeurons, HiddenNeurons, OutputNeurons), optimizer(nn, opt, Epoches, LFN){

        this->taskType = TT;
        this->learningMethod = LRM;

    }

    AMGNN(std::vector<std::vector<int>> kernelsInfos, std::vector<std::vector<int>> poolingsInfos, std::vector<int> padding, ActivationFunction CAFN, int channels, bool GAP, std::vector<int> InputSize, ActivationFunction DAFN, std::vector<int> HiddenNeurons, int outputNeurons, TaskType taskType, Loss_Function LFN, Optimizers opt, int Epoches, LearningMethod LRM)
    : cnn(kernelsInfos, poolingsInfos, padding, CAFN, channels, GAP, InputSize, DAFN, HiddenNeurons, outputNeurons, taskType), optimizer(cnn, opt, Epoches, LFN){

        this->taskType = taskType;
        this->learningMethod = LRM;

    }

    AMGNN(){}

    double max(std::vector<double> Input){
        double maxim = Input[0];
        for(double value:Input){
            if(value > maxim){
                maxim = value;
            }
        }
        return maxim;
    }

    void argmax(std::vector<double>& Input){
        double maximaum = max(Input);
        for(int i = 0; i < Input.size(); i++){
            Input[i] = (Input[i] < maximaum)?0:1;
        }
    }

    void NormB_MLClassification(std::vector<double>& Input){
        for(int i = 0; i < Input.size(); i++){
            Input[i] = (Input[i] < 0.5)?0:1;
        }
    }

public:
    template<class INPUT>
    void train(std::vector<INPUT>& InputData, std::vector<std::vector<double>>& OutputData){
        switch(taskType){
            case REGRESSION:
            case BINARRY_CLASSIFICATION:
            case MULTICLASS_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION : 
                switch(learningMethod){
                    case GRADIENT_DESCENT : optimizer.gradientDescent(nn, InputData, OutputData); break;
                    case STOCHASTIC_GRADIENT_DESCENT : optimizer.stochasticGradientDescent(nn, InputData, OutputData); break;
                    case MINI_BATCH_GRADIENT_DESCENT : optimizer.miniBatchGradientDescent(nn, InputData, OutputData); break;
                    case LBFGS_Method : optimizer.LBFGS_Optimizing(nn, InputData, OutputData); break;
                }
                break;
            case BINARRY_IMAGE_CLASSIFICATION:
            case MULTICLASS_IMAGE_CLASSIFICATION:
            case MULTILABEL_IMAGE_CLASSIFICATION :
                switch(learningMethod){
                    case GRADIENT_DESCENT : optimizer.gradientDescent(cnn, InputData, OutputData); break;
                    case STOCHASTIC_GRADIENT_DESCENT : optimizer.stochasticGradientDescent(cnn, InputData, OutputData); break;
                    case MINI_BATCH_GRADIENT_DESCENT : optimizer.miniBatchGradientDescent(cnn, InputData, OutputData); break;
                    case LBFGS_Method : optimizer.LBFGS_Optimizing(cnn, InputData, OutputData); break;
                }
                break;
        }
    }

    template<class INPUT>
    std::vector<double> predict(INPUT Input){
        std::vector<double> predicted;

        switch(taskType){
            case REGRESSION:
            case BINARRY_CLASSIFICATION:
            case MULTICLASS_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION : predicted = nn.predict(Input); break;
            case BINARRY_IMAGE_CLASSIFICATION:
            case MULTICLASS_IMAGE_CLASSIFICATION:
            case MULTILABEL_IMAGE_CLASSIFICATION : predicted = cnn.predict(Input);
        }

        switch(taskType){
            case MULTICLASS_CLASSIFICATION: argmax(predicted); break;
            case BINARRY_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION: NormB_MLClassification(predicted); break;
        }
        return predicted;
    }

private:
    void load(std::ifstream& fileIn){

        int ttype;
        fileIn.read(reinterpret_cast<char*>(&ttype), sizeof(int));
        taskType = static_cast<TaskType>(ttype);

        int LM;
        fileIn.read(reinterpret_cast<char*>(&LM), sizeof(int));
        learningMethod = static_cast<LearningMethod>(LM);

        switch(taskType){
            case REGRESSION:
            case BINARRY_CLASSIFICATION:
            case MULTICLASS_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION : nn.load(fileIn, true); break;
            case BINARRY_IMAGE_CLASSIFICATION:
            case MULTICLASS_IMAGE_CLASSIFICATION:
            case MULTILABEL_IMAGE_CLASSIFICATION : cnn.load(fileIn); break;
        }

        optimizer.load(fileIn);

    }

public:
    void saveModel(std::string fileName){

        std::ofstream fileOut(fileName+".AMGNN", std::ios::binary);

        int ttype = static_cast<int>(taskType);
        fileOut.write(reinterpret_cast<char*>(&ttype), sizeof(int));

        int LM = static_cast<int>(learningMethod);
        fileOut.write(reinterpret_cast<char*>(&LM), sizeof(int));

        switch(taskType){
            case REGRESSION:
            case BINARRY_CLASSIFICATION:
            case MULTICLASS_CLASSIFICATION:
            case MULTILABEL_CLASSIFICATION : nn.save(fileOut); break;
            case BINARRY_IMAGE_CLASSIFICATION:
            case MULTICLASS_IMAGE_CLASSIFICATION:
            case MULTILABEL_IMAGE_CLASSIFICATION : cnn.save(fileOut); break;
        }

        optimizer.save(fileOut);

    }

    static AMGNN loadModel(std::string fileName){

        std::ifstream fileIn(fileName, std::ios::binary);

        AMGNN neuralnetrwork;
        neuralnetrwork.load(fileIn);

        return neuralnetrwork;

    }

};