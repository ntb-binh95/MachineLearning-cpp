#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>


using namespace std;

#define LOG(x) cout << x << endl

class LogisticRegression {
    public:
        LogisticRegression(){};
        LogisticRegression(int input_feats, float learning_rate, int n_iters): 
        lr{learning_rate}, n_iters{n_iters}, input_features{input_feats} {
            initParams();
        };
        
        void fit(unique_ptr<float[]>& X, unique_ptr<float[]>& y, size_t samples) {
            for (int i = 0; i < n_iters; i++){
                // y_hat = getPrediction(X.get());
            }
        }

    protected:
        void initParams() {
            weights = make_unique<float[]>(input_features);
            bias = 0;
        }

        float getPrediction(unique_ptr<float[]> input) {
            // output = input * weight + bias;
            float output = 0;
            for(int i = 0; i < input_features; i++){
                output += input[i] * weights[i];
            }
            output += bias;

            // sigmoid
            output == sigmoid(output);
            return output;
        }
    
    private:
        float lr = 0.001;
        int n_iters = 1000;
        int input_features;
        unique_ptr<float[]> weights;
        float bias = 0;
        float sigmoid(float x){
            return 1 / (1 + exp(-x));
        }
};

class BreastCancerDataset {
    public:
        BreastCancerDataset() {
            string filePath = "wdbc.data";

            ifstream file{filePath};

            if(!file.is_open()){
                LOG("Can not open dataset file");
            }

            string line = "";
            string ID = "";
            string label = "";
            string value = "";
            while(getline(file, line)){
                stringstream ss(line);
                getline(ss, ID, ',');
                getline(ss, label, ',');
                vector<float> feature;
                while(getline(ss, value, ',')){
                    feature.push_back(stof(value));
                }
                features.push_back(feature);
                if(label == "B") {
                    ground_truth.push_back(0);
                }
                else {
                    ground_truth.push_back(1);
                }
            }
            assert(features.size() == ground_truth.size());
        };
    
    private:
        float train_test_split = 0.2;
        vector<vector<float>> features;
        vector<float> ground_truth;
};

int main(int argc, char** argv){
    /*
    TODO:
    1. Add breast cancer dataset (DONE)
    2. Add logistic regression class
    3. Implement matrix multiplication
    4. add shuffle to dataset
    */
    BreastCancerDataset  dataset;

    return 0;
}