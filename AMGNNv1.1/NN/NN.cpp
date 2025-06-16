#include"OutputLayer.cpp"

class NN{

public:
    std::vector<HiddenLayer> HiddenLayers;
    OutputLayer outputLayer;


    int MemoryLayer = 0;
    bool MemoryW_B = false;
    int MemoryPOS = 0;
    bool MemoryOutput = false;
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

    }

    NN(){}

    void changeValue(int layer, int pos, bool W_B, bool output, double value){

        if(output){
            if(W_B){
                MemoryValue = outputLayer.InWeights[pos];
                outputLayer.InWeights[pos] = value;
            }else{
                MemoryValue = outputLayer.biases[pos];
                outputLayer.biases[pos] = value;
            }
        }else{
            if(W_B){
                MemoryValue = HiddenLayers[layer].InWeights[pos];
                HiddenLayers[layer].InWeights[pos] = value;
            }else{
                MemoryValue = HiddenLayers[layer].biases[pos];
                HiddenLayers[layer].biases[pos] = value;
            }
        }

        MemoryLayer = layer;
        MemoryW_B = W_B;
        MemoryPOS = pos;
        MemoryOutput = output;

    }

    void restoreValues(){

        if(MemoryOutput){
            if(MemoryW_B){
                outputLayer.InWeights[MemoryPOS] = MemoryValue;
            }else{
                outputLayer.biases[MemoryPOS] = MemoryValue;
            }
        }else{
            if(MemoryW_B){
                HiddenLayers[MemoryLayer].InWeights[MemoryPOS] = MemoryValue;
            }else{
                HiddenLayers[MemoryLayer].biases[MemoryPOS] = MemoryValue;
            }
        }

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