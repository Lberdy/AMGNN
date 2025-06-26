#include<vector>
#include<random>
#include<cmath>
#include<fstream>
#include<iostream>

class Matrix{

public:

    static std::vector<double> zeros(int e){

        std::vector<double> zero;
        for(int i = 1; i <= e; i++){
            zero.push_back(0);
        }

        return zero;

    }

    static std::vector<double> Xavier(int fanIn, int fanOut){

        double limit = sqrt(6.0/(fanIn+fanOut));
        std::uniform_real_distribution<double> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<double> xavier;
        for(int i = 1; i <= fanOut*fanIn; i++){
            xavier.push_back(destribution(engine));
        }

        return xavier;

    }

    static std::vector<std::vector<double>> Xavier2D(int fanIn, int fanOut, int row, int col){

        double limit = sqrt(6.0/(fanIn+fanOut));
        std::uniform_real_distribution<double> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<std::vector<double>> xavier;
        for(int i = 0; i < row; i++){
            std::vector<double> xavierRow;
            for(int j = 0; j < col; j++){
                xavierRow.push_back(destribution(engine));
            }
            xavier.push_back(xavierRow);
        }

        return xavier;

    }


    static std::vector<double> Kaiming(int fanIn, int fanOut){

        double limit = sqrt(6.0/(fanIn));
        
        std::uniform_real_distribution<double> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<double> kaiming;
        for(int i = 1; i <= fanOut*fanIn; i++){
            kaiming.push_back(destribution(engine));
        }

        return kaiming;

    }

    static std::vector<std::vector<double>> Kaiming2D(int fanIn, int fanOut, int row, int col){

        double limit = sqrt(6.0/(fanIn+fanOut));
        std::uniform_real_distribution<double> destribution(-limit,limit);
        std::random_device rd;
        std::default_random_engine engine(rd());

        std::vector<std::vector<double>> kaiming;
        for(int i = 0; i < row; i++){
            std::vector<double> kaimingRow;
            for(int j = 0; j < col; j++){
                kaimingRow.push_back(destribution(engine));
            }
            kaiming.push_back(kaimingRow);
        }

        return kaiming;

    }

    static void dot(std::vector<double> mat1, std::vector<double> mat2, std::vector<double>& values){

        int i = 0;
        int index = 0;
        int size = mat2.size();
        double sum = 0;
        while(i < mat1.size()){
            sum += mat1[i]*mat2[i%size];
            i++;
            if(i % size == 0){
                values[index] = sum;
                index++;
                sum = 0;
            }
        }

    }

    static double fastDot(std::vector<double>& line1, std::vector<double>& line2){
        double sum = 0;
        for(int i = 0; i < line1.size(); i++){
            sum += line1[i]*line2[i];
        }
        return sum;
    }

};