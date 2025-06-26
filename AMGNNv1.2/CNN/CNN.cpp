#include"ConvolutionLayer.cpp"

class CNN{

public:
    std::vector<ConvolutionLayer> ConvolutionLayers;
    NN DenseLayer;
    std::vector<double*> FlattenWeights;
    bool GloabalAveragePooling;

    int MemoryIndex = 0;
    double MemoryValue = 0;

    void initializeFlattenWeights(){

        for(ConvolutionLayer& layer:ConvolutionLayers){
            for(std::vector<std::vector<std::vector<double>>>& kernel:layer.Kernels){
                for(std::vector<std::vector<double>>& subKerenl:kernel){
                    for(std::vector<double>& subKernelRow:subKerenl){
                        for(int i = 0; i < subKernelRow.size(); i++){
                            FlattenWeights.push_back(&subKernelRow[i]);
                        }
                    }
                }
            }
            for(int i = 0; i < layer.Biases.size(); i++){
                FlattenWeights.push_back(&layer.Biases[i]);
            }
        }

        for(HiddenLayer& layer:DenseLayer.HiddenLayers){
            for(int i = 0; i < layer.InWeights.size(); i++){
                FlattenWeights.push_back(&layer.InWeights[i]);
            }
            for(int i = 0; i < layer.biases.size(); i++){
                FlattenWeights.push_back(&layer.biases[i]);
            }
        }
        for(int i = 0; i < DenseLayer.outputLayer.InWeights.size(); i++){
            FlattenWeights.push_back(&DenseLayer.outputLayer.InWeights[i]);
        }
        for(int i = 0; i < DenseLayer.outputLayer.biases.size(); i++){
            FlattenWeights.push_back(&DenseLayer.outputLayer.biases[i]);
        }

    }

    CNN(std::vector<std::vector<int>> kernelsInfos, std::vector<std::vector<int>> poolingsInfos, std::vector<int> padding, ActivationFunction CAFN, int channels, bool GAP, std::vector<int> InputSize, ActivationFunction DAFN, std::vector<int> HiddenNeurons, int outputNeurons, TaskType taskType){

        GloabalAveragePooling = GAP;

        for(int i = 0; i < kernelsInfos.size(); i++){
            if(i == 0){
                ConvolutionLayer layer = ConvolutionLayer(kernelsInfos[i], CAFN, poolingsInfos[i], padding[i], channels, InputSize);
                ConvolutionLayers.push_back(layer);
            }else{
                ConvolutionLayer layer = ConvolutionLayer(kernelsInfos[i], CAFN, poolingsInfos[i], padding[i], ConvolutionLayers[i-1].KernelInfos[0], ConvolutionLayers[i-1].FeatureSize);
                ConvolutionLayers.push_back(layer);
            }
        }

        if(GloabalAveragePooling){
            DenseLayer.init(DAFN, taskType, ConvolutionLayers.back().KernelInfos[0], HiddenNeurons, outputNeurons);
        }else{
            int featureHeight = ConvolutionLayers.back().FeatureSize[0];
            int featureWidth = ConvolutionLayers.back().FeatureSize[1];
            int featuresNumber = ConvolutionLayers.back().KernelInfos[0];
            DenseLayer.init(DAFN, taskType, featureHeight*featureWidth*featuresNumber, HiddenNeurons, outputNeurons);
        }

        initializeFlattenWeights();

    }

    CNN(){}

    CNN deepCopy(CNN& nn){

        CNN cnnCopy = nn;

        cnnCopy.FlattenWeights.clear();

        for(ConvolutionLayer& layer:cnnCopy.ConvolutionLayers){
            for(std::vector<std::vector<std::vector<double>>>& kernel:layer.Kernels){
                for(std::vector<std::vector<double>>& subKerenl:kernel){
                    for(std::vector<double>& subKernelRow:subKerenl){
                        for(int i = 0; i < subKernelRow.size(); i++){
                            FlattenWeights.push_back(&subKernelRow[i]);
                        }
                    }
                }
            }
            for(int i = 0; i < layer.Biases.size(); i++){
                FlattenWeights.push_back(&layer.Biases[i]);
            }
        }

        for(HiddenLayer& layer:cnnCopy.DenseLayer.HiddenLayers){
            for(int i = 0; i < layer.InWeights.size(); i++){
                FlattenWeights.push_back(&layer.InWeights[i]);
            }
            for(int i = 0; i < layer.biases.size(); i++){
                FlattenWeights.push_back(&layer.biases[i]);
            }
        }
        for(int i = 0; i < cnnCopy.DenseLayer.outputLayer.InWeights.size(); i++){
            FlattenWeights.push_back(&cnnCopy.DenseLayer.outputLayer.InWeights[i]);
        }
        for(int i = 0; i < cnnCopy.DenseLayer.outputLayer.biases.size(); i++){
            FlattenWeights.push_back(&cnnCopy.DenseLayer.outputLayer.biases[i]);
        }


        return cnnCopy;

    }

    void changeValue(int index, double value){
        MemoryIndex = index;
        MemoryValue = value;
        *FlattenWeights[index] = value;
    }

    void restoreValue(){
        *FlattenWeights[MemoryIndex] = MemoryValue;
    }

private:
    std::vector<double> flatting(std::vector<std::vector<std::vector<double>>>& features){

        std::vector<double> flat;

        if(GloabalAveragePooling){
            for(std::vector<std::vector<double>>& feature:features){
                double sum = 0;
                for(std::vector<double>& featureRow:feature){
                    for(double& value:featureRow){
                        sum += value;
                    }
                }
                flat.push_back(sum/(features[0].size()*features[0][0].size()));
            }
        }else{
            for(std::vector<std::vector<double>>& feature:features){
                for(std::vector<double>& featureRow:feature){
                    for(double& value:featureRow){
                        flat.push_back(value);
                    }
                }
            }
        }

        return flat;

    }

public:
    std::vector<double> predict(cv::Mat img){

        for(int i = 0; i < ConvolutionLayers.size(); i++){
            if(i == 0){
                ConvolutionLayers[i].calculateFeatures(img);
            }else{
                ConvolutionLayers[i].calculateFeatures(ConvolutionLayers[i-1].Features);
            }
        }


        return DenseLayer.predict(flatting(ConvolutionLayers.back().Features));

    }

    void save(std::ofstream& fileOut){

        size_t layersNumber = ConvolutionLayers.size();
        fileOut.write(reinterpret_cast<char*>(&layersNumber), sizeof(layersNumber));
        for(size_t i = 0; i < layersNumber; i++){
            ConvolutionLayers[i].save(fileOut);
        }

        DenseLayer.save(fileOut);

        fileOut.write(reinterpret_cast<char*>(GloabalAveragePooling), sizeof(bool));

    }

    void load(std::ifstream& fileIn){

        size_t layersNumber;
        fileIn.read(reinterpret_cast<char*>(&layersNumber), sizeof(layersNumber));
        for(size_t i = 0; i < layersNumber; i++){
            ConvolutionLayer layer;
            layer.load(fileIn);
            ConvolutionLayers.push_back(layer);
        }

        DenseLayer.load(fileIn, false);

        initializeFlattenWeights();

        fileIn.read(reinterpret_cast<char*>(&GloabalAveragePooling), sizeof(bool));

    }

};