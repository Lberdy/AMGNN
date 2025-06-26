#include"../NN/NN.cpp"
#include<opencv2/opencv.hpp>

enum Poolings{
    MEAN_POOLING,
    MAX_POOLING
};

class ConvolutionLayer{

public:
    std::vector<std::vector<std::vector<std::vector<double>>>> Kernels;
    std::vector<double> Biases;
    std::vector<std::vector<std::vector<double>>> Features;
private:
    ActivationFunction ActivationFunctionName;
public:
    std::vector<int> KernelInfos;
private:
    std::vector<int> PoolingInfos;
public:
    std::vector<int> FeatureSize;
    Poolings PoolingName = MAX_POOLING;
    double paddingPixelValue = 0;
private:
    int padding;

    void calculateFeatureSize(std::vector<int> InputSIze){

        if(InputSIze.size() > 0){
            int featureHeightWP = 1;
            int featureWidthWP = 1;

            for(int i = 0; i + KernelInfos[1] < InputSIze[0] + padding*2; i += KernelInfos[3]){
                featureHeightWP++;
            }
            for(int i = 0; i + KernelInfos[2] < InputSIze[1] + padding*2; i += KernelInfos[3]){
                featureWidthWP++;
            }

            if(PoolingInfos.size() > 0){
                int featureHeight = 1;
                int featureWidth = 1;

                for(int i = 0; i + PoolingInfos[0] < featureHeightWP; i += PoolingInfos[2]){
                    featureHeight++;
                }
                for(int i = 0; i + PoolingInfos[1] < featureWidthWP; i += PoolingInfos[2]){
                    featureWidth++;
                }

                FeatureSize.push_back(featureHeight);
                FeatureSize.push_back(featureWidth);

            }else{
                FeatureSize.push_back(featureHeightWP);
                FeatureSize.push_back(featureWidthWP);
            }

        }

    }

public:
    ConvolutionLayer(std::vector<int> KInfos, ActivationFunction AFN, std::vector<int> PInfos, int padd, int channels, std::vector<int> InputSize){

        for(int i = 0; i < KInfos[0]; i++){
            std::vector<std::vector<std::vector<double>>> kernel;
            for(int j = 0; j < channels; j++){
                std::vector<std::vector<double>> subKernel;
                if(AFN == LINEAR || AFN == SIGMOID || AFN == TANH){
                    subKernel = Matrix::Xavier2D(channels*KInfos[1]*KInfos[2], KInfos[0]*KInfos[1]*KInfos[2], KInfos[1], KInfos[2]);
                }else if(AFN == RELU || AFN == SOFTPLUS){
                    subKernel = Matrix::Kaiming2D(channels*KInfos[1]*KInfos[2], KInfos[0]*KInfos[1]*KInfos[2], KInfos[1], KInfos[2]);
                }
                kernel.push_back(subKernel);
            }
            Kernels.push_back(kernel);
        }

        Biases = Matrix::zeros(KInfos[0]);

        ActivationFunctionName = AFN;
        KernelInfos = KInfos;
        PoolingInfos = PInfos;
        padding = padd;

        calculateFeatureSize(InputSize);

    }

    ConvolutionLayer(){}

private:
    double Linear(double x){
        return x;
    }

    double sigmoid(double x){
        return 1.0/(1 + exp(-x));
    }

    double Tanh(double x){
        return tanh(x);
    }

    double relu(double x){
        return (x > 0)?x:0;
    }

    double softplus(double x){
        return log(1 + exp(x));
    }

    double max(std::vector<double> values){
        double maxim = values[0];
        for(double val:values){
            if(val > maxim){
                maxim = val;
            }
        }
        return maxim;
    }

    double mean(std::vector<double> values){
        double sum = 0;
        for(double val:values){
            sum += val;
        }
        return sum/(double)values.size();
    }

    std::vector<std::vector<double>> Pooling(std::vector<std::vector<double>> feature){

        std::vector<std::vector<double>> featureAP;

        for(int i = 0; i + PoolingInfos[0] < FeatureSize[0]; i += PoolingInfos[2]){
            std::vector<double> featureRow;
            for(int j = 0; j + PoolingInfos[1] < FeatureSize[1]; j += PoolingInfos[2]){
                std::vector<double> values;
                for(int h = i; h < i + PoolingInfos[0]; h++){
                    for(int w = j; w < j + PoolingInfos[1]; w++){
                        values.push_back(feature[h][w]);
                    }
                }
                double cellValue;
                switch(PoolingName){
                    case MAX_POOLING : cellValue = max(values); break;
                    case MEAN_POOLING : cellValue = mean(values); break;
                }
                featureRow.push_back(cellValue);
            }
            featureAP.push_back(featureRow);
        }

        return featureAP;

    }

    std::vector<double> getPixels(cv::Mat img, int row, int col){

        std::vector<double> pixels;

        if(img.channels() == 1){
            pixels.push_back(img.at<double>(row, col));
            return pixels;
        }

        double* imgRow = img.ptr<double>(row);
        for(int i = 0; i < img.channels(); i++){
            pixels.push_back(imgRow[col*img.channels() + i]);
        }

        return pixels;

    }

public:
    void calculateFeatures(cv::Mat img){

        Features.clear();

        for(int k = 0; k < Kernels.size(); k++){

            std::vector<std::vector<std::vector<double>>> kernel = Kernels[k];
            std::vector<std::vector<double>> feature;

            for(int h = 0; h < img.rows + padding*2; h++){
                std::vector<double> featureRow;
                for(int w = 0; w < img.cols + padding*2; w++){
                    int imageX = w;
                    int imageY = h;
                    double sum = 0;
                    for(int i = 0; i < KernelInfos[1]; i++){
                        for(int j = 0; j < KernelInfos[2]; j++){
                            std::vector<double> pixels;
                            if(imageX >= padding && imageX < img.cols + padding && imageY >= padding && imageY < img.rows + padding){
                                pixels = getPixels(img, imageY, imageX);
                            }
                            for(int n = 0; n < kernel.size(); n++){
                                if(imageX >= padding && imageX < img.cols + padding && imageY >= padding && imageY < img.rows + padding){
                                    sum += kernel[n][i][j]*pixels[n];
                                }else{
                                    sum += kernel[n][i][j]*paddingPixelValue;
                                }
                            }
                            imageX++;
                        }
                        imageX = w;
                        imageY++;
                    }
                    sum += Biases[k];
                    switch(ActivationFunctionName){
                        case LINEAR : sum = Linear(sum); break;
                        case SIGMOID : sum = sigmoid(sum); break;
                        case TANH : sum = Tanh(sum); break;
                        case RELU : sum = relu(sum); break;
                        case SOFTPLUS : sum = softplus(sum); break;
                    }
                    featureRow.push_back(sum);
                }
                feature.push_back(featureRow);
            }
            if(PoolingInfos.size() > 0){
                feature = Pooling(feature);
            }
            Features.push_back(feature);
        }

    }

    void calculateFeatures(std::vector<std::vector<std::vector<double>>> features){

        Features.clear();

        for(int k = 0; k < Kernels.size(); k++){

            std::vector<std::vector<std::vector<double>>> kernel = Kernels[k];
            std::vector<std::vector<double>> feature;

            for(int h = 0; h < features[0].size() + padding*2; h++){
                std::vector<double> featureRow;
                for(int w = 0; w < features[0][0].size() + padding*2; w++){
                    int imageX = w;
                    int imageY = h;
                    double sum = 0;
                    for(int i = 0; i < KernelInfos[1]; i++){
                        for(int j = 0; j < KernelInfos[2]; j++){
                            for(int n = 0; n < kernel.size(); n++){
                                if(imageX >= padding && imageX < features[0][0].size() + padding && imageY >= padding && imageY < features[0].size() + padding){
                                    sum += kernel[n][i][j]*features[n][i][j];
                                }else{
                                    sum += kernel[n][i][j]*paddingPixelValue;
                                }
                            }
                            imageX++;
                        }
                        imageX = w;
                        imageY++;
                    }
                    sum += Biases[k];
                    switch(ActivationFunctionName){
                        case LINEAR : sum = Linear(sum); break;
                        case SIGMOID : sum = sigmoid(sum); break;
                        case TANH : sum = Tanh(sum); break;
                        case RELU : sum = relu(sum); break;
                        case SOFTPLUS : sum = softplus(sum); break;
                    }
                    featureRow.push_back(sum);
                }
                feature.push_back(featureRow);
            }
            if(PoolingInfos.size() > 0){
                feature = Pooling(feature);
            }
            Features.push_back(feature);
        }

    }

    void save(std::ofstream& fileOut){

        size_t kernelNumber = Kernels.size();
        fileOut.write(reinterpret_cast<char*>(&kernelNumber), sizeof(kernelNumber));
        for(size_t i = 0; i < kernelNumber; i++){
            std::vector<std::vector<std::vector<double>>> kernel = Kernels[i];
            size_t channels = kernel.size();
            fileOut.write(reinterpret_cast<char*>(&channels), sizeof(channels));
            for(size_t j = 0; j < channels; j++){
                std::vector<std::vector<double>> subKernel = kernel[j];
                size_t kernelHeight = subKernel.size();
                fileOut.write(reinterpret_cast<char*>(&kernelHeight), sizeof(kernelHeight));
                for(int h = 0; h < kernelHeight; h++){
                    std::vector<double> kernelRow = subKernel[h];
                    size_t kernelWidth = kernelRow.size();
                    fileOut.write(reinterpret_cast<char*>(&kernelWidth), sizeof(kernelWidth));
                    for(size_t w = 0; w < kernelWidth; w++){
                        double weight = kernelRow[w];
                        fileOut.write(reinterpret_cast<char*>(&weight), sizeof(double));
                    }
                }
            }
        }

        size_t BiasesSize = Biases.size();
        fileOut.write(reinterpret_cast<char*>(&BiasesSize), sizeof(BiasesSize));
        for(size_t i = 0; i < BiasesSize; i++){
            double bias = Biases[i];
            fileOut.write(reinterpret_cast<char*>(&bias), sizeof(double));
        }

        int AFN = static_cast<int>(ActivationFunctionName);
        fileOut.write(reinterpret_cast<char*>(&AFN), sizeof(int));

        size_t kernelInfosSize = KernelInfos.size();
        fileOut.write(reinterpret_cast<char*>(&kernelInfosSize), sizeof(kernelInfosSize));
        for(size_t i = 0; i < kernelInfosSize; i++){
            int info = KernelInfos[i];
            fileOut.write(reinterpret_cast<char*>(&info), sizeof(int));
        }

        size_t PoolingInfosSize = PoolingInfos.size();
        fileOut.write(reinterpret_cast<char*>(&PoolingInfosSize), sizeof(PoolingInfosSize));
        for(size_t i = 0; i < PoolingInfosSize; i++){
            int info = PoolingInfos[i];
            fileOut.write(reinterpret_cast<char*>(&info), sizeof(int));
        }

        int featureHeight = FeatureSize[0];
        int featureWidth = FeatureSize[1];
        fileOut.write(reinterpret_cast<char*>(&featureHeight), sizeof(int));
        fileOut.write(reinterpret_cast<char*>(&featureWidth), sizeof(int));

        int PN = static_cast<int>(PoolingName);
        fileOut.write(reinterpret_cast<char*>(&PN), sizeof(int));

        fileOut.write(reinterpret_cast<char*>(&paddingPixelValue), sizeof(double));

        fileOut.write(reinterpret_cast<char*>(&padding), sizeof(int));

    }

    void load(std::ifstream& fileIn){

        size_t kernelNumber;
        fileIn.read(reinterpret_cast<char*>(&kernelNumber), sizeof(kernelNumber));
        for(size_t i = 0; i < kernelNumber; i++){
            std::vector<std::vector<std::vector<double>>> kernel;
            size_t channels;
            fileIn.read(reinterpret_cast<char*>(&channels), sizeof(channels));
            for(size_t j = 0; j < channels; j++){
                std::vector<std::vector<double>> subKernel;
                size_t kernelHeight;
                fileIn.read(reinterpret_cast<char*>(&kernelHeight), sizeof(kernelHeight));
                for(int h = 0; h < kernelHeight; h++){
                    std::vector<double> kernelRow;
                    size_t kernelWidth;
                    fileIn.read(reinterpret_cast<char*>(&kernelWidth), sizeof(kernelWidth));
                    for(size_t w = 0; w < kernelWidth; w++){
                        double weight;
                        fileIn.read(reinterpret_cast<char*>(&weight), sizeof(weight));
                        kernelRow.push_back(weight);
                    }
                    subKernel.push_back(kernelRow);
                }
                kernel.push_back(subKernel);
            }
            Kernels.push_back(kernel);
        }

        size_t BiasesSize;
        fileIn.read(reinterpret_cast<char*>(&BiasesSize), sizeof(BiasesSize));
        for(size_t i = 0; i < BiasesSize; i++){
            double weight;
            fileIn.read(reinterpret_cast<char*>(&weight), sizeof(double));
            Biases.push_back(weight);
        }

        int AFN;
        fileIn.read(reinterpret_cast<char*>(&AFN), sizeof(int));
        ActivationFunctionName = static_cast<ActivationFunction>(AFN);

        size_t KernelInfosSize;
        fileIn.read(reinterpret_cast<char*>(&KernelInfosSize), sizeof(KernelInfosSize));
        for(size_t i = 0; i < KernelInfosSize; i++){
            int info;
            fileIn.read(reinterpret_cast<char*>(&info), sizeof(int));
            KernelInfos.push_back(info);
        }

        size_t PoolingInfosSize;
        fileIn.read(reinterpret_cast<char*>(&PoolingInfosSize), sizeof(PoolingInfosSize));
        for(size_t i = 0; i < PoolingInfosSize; i++){
            int info;
            fileIn.read(reinterpret_cast<char*>(&info), sizeof(int));
            PoolingInfos.push_back(info);
        }

        int featureHeight;
        int featureWidth;
        fileIn.read(reinterpret_cast<char*>(&featureHeight), sizeof(int));
        fileIn.read(reinterpret_cast<char*>(&featureWidth), sizeof(int));

        int PN;
        fileIn.read(reinterpret_cast<char*>(&PN), sizeof(int));
        PoolingName = static_cast<Poolings>(PN);

        fileIn.read(reinterpret_cast<char*>(&paddingPixelValue), sizeof(double));

        fileIn.read(reinterpret_cast<char*>(&padding), sizeof(int));

    }

};