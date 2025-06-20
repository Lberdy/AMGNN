#include"OutputLayer.cpp"

class NN{

public:
    std::vector<HiddenLayer> HiddenLayers;
    OutputLayer outputLayer;
    std::vector<double*> FlattenWeights;


    int MemoryIndex = 0;
    double MemoryValue = 0;

public:
    NN(ActivationFunction AFN, TaskType TT, int InputNeurons, std::vector<int> HiddenNeurons, int OutputNeurons)
    : outputLayer(TT, HiddenNeurons[HiddenNeurons.size()-1], OutputNeurons) {

        for(int i = 0; i < HiddenNeurons.size(); i++){
            if(i == 0){
                HiddenLayer hiddenlayer(AFN, InputNeurons, HiddenNeurons[i]);
                HiddenLayers.push_back(hiddenlayer);
            }else{
                HiddenLayer hiddenlayer(AFN, HiddenNeurons[i-1], HiddenNeurons[i]);
                HiddenLayers.push_back(hiddenlayer);
            }
        }

        for(HiddenLayer& layer:HiddenLayers){
            for(int i = 0; i < layer.InWeights.size(); i++){
                FlattenWeights.push_back(&layer.InWeights[i]);
            }
            for(int i = 0; i < layer.biases.size(); i++){
                FlattenWeights.push_back(&layer.biases[i]);
            }
        }
        for(int i = 0; i < outputLayer.InWeights.size(); i++){
            FlattenWeights.push_back(&outputLayer.InWeights[i]);
        }
        for(int i = 0; i < outputLayer.biases.size(); i++){
            FlattenWeights.push_back(&outputLayer.biases[i]);
        }

    }

    NN(){}

    NN deepCopy(NN& nn){
        NN nnCopy = nn;
        nnCopy.FlattenWeights.clear();

        for(HiddenLayer& layer:nnCopy.HiddenLayers){
            for(int i = 0; i < layer.InWeights.size(); i++){
                nnCopy.FlattenWeights.push_back(&layer.InWeights[i]);
            }
            for(int i = 0; i < layer.biases.size(); i++){
                nnCopy.FlattenWeights.push_back(&layer.biases[i]);
            }
        }
        for(int i = 0; i < nnCopy.outputLayer.InWeights.size(); i++){
            nnCopy.FlattenWeights.push_back(&nnCopy.outputLayer.InWeights[i]);
        }
        for(int i = 0; i < nnCopy.outputLayer.biases.size(); i++){
            nnCopy.FlattenWeights.push_back(&nnCopy.outputLayer.biases[i]);
        }

        return nnCopy;

    }

    void changeValue(int Index, double value){

        MemoryIndex = Index;
        MemoryValue = *FlattenWeights[Index];
        *FlattenWeights[Index] = value;

    }

    void restoreValue(){

        *FlattenWeights[MemoryIndex] = MemoryValue;

    }

    std::vector<double> predict(std::vector<double> Input){
        for(int i = 0; i < HiddenLayers.size(); i++){

            HiddenLayer& layer = HiddenLayers[i];
            
            if(i == 0){
                layer.calculateValues(Input);
            }else{
                layer.calculateValues(HiddenLayers[i-1].values);
            }
        }

        outputLayer.calculateValues(HiddenLayers[HiddenLayers.size()-1].values);

        return outputLayer.values;

    }

    void save(std::ofstream& fileOut){

        size_t hiddenLayersSize = HiddenLayers.size();
        fileOut.write(reinterpret_cast<char*>(&hiddenLayersSize), sizeof(hiddenLayersSize));

        for(size_t i = 0; i < hiddenLayersSize; i++){
            HiddenLayer& layer = HiddenLayers[i];
            layer.save(fileOut);
        }

        outputLayer.save(fileOut);

    }

    void load(std::ifstream& fileIn){

        size_t hiddenLayersSize;
        fileIn.read(reinterpret_cast<char*>(&hiddenLayersSize), sizeof(hiddenLayersSize));

        for(size_t i = 0; i < hiddenLayersSize; i++){
            HiddenLayer layer;
            layer.load(fileIn);
            HiddenLayers.push_back(layer);
        }

        outputLayer.load(fileIn);

    }

};