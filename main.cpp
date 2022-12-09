#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

#include "gemm.h"

using namespace std;

#define LOG(x) cout << x << endl

float rand_uniform(float min, float max) {
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

class LogisticRegression {
    public:
        LogisticRegression(int input_feats): input_features{input_feats} {};
        LogisticRegression(int input_feats, float learning_rate, int n_iters): 
        lr{learning_rate}, n_iters{n_iters}, input_features{input_feats} {};
        
        void fit(unique_ptr<float[]>& X, unique_ptr<float[]>& y, int samples) {
            initParams();
            for (int i = 0; i < n_iters; i++){
                unique_ptr<float[]> y_pred = getPrediction(X, samples);

                computeGradient(X, y, y_pred, samples);
                updateParams();
            }
        }

        unique_ptr<float[]> predict(unique_ptr<float[]>& X, int samples){
            unique_ptr<float[]> out = getPrediction(X, samples);
            unique_ptr<float[]> ret = make_unique<float[]>(samples);
            float threshold = 0.5;
            for(int i = 0; i < samples; i++){
                ret[i] = out[i] > threshold ? 1 : 0;
            }
            return ret;
        }

        float accuracy(unique_ptr<float[]> &y_gt, unique_ptr<float[]> &y_pred, int samples){
            int error_count = 0;
            for (int i = 0; i < samples; i++) {
                if(y_gt[i] != y_pred[i]){
                    error_count++;
                }
            }
            return error_count * 100.f / samples;
        }

    protected:
        void initParams() {
            weights = make_unique<float[]>(input_features);
            // random initialize the weight
            float scale = sqrt(2./input_features);
            for (int i = 0; i < input_features; i++){
                weights[i] = scale * rand_uniform(-1, 1);
            }

            dW = make_unique<float[]>(input_features);
            bias = 0;
            dB = 0;
        }

        unique_ptr<float[]> getPrediction(unique_ptr<float[]> &input, int samples) {
            // output = input * weight + bias;
            // input (samples x feats)   | a(m,k)
            // weights (feats x 1)         | b(k,n)
            // output (samples x 1)      | c(m,n)
            unique_ptr<float[]> output = make_unique<float[]>(samples);
            int m = samples;
            int k = input_features;
            int n = 1;
            float * a = input.get();
            float * b = weights.get();
            float * c = output.get();
            gemm(0, 0, m, n, k, 1, a, k, b, n, 0, c, n);

            // add bias, sigmoid
            for(int i = 0; i < samples; i++) {
                output[i] += bias;
                output[i] = sigmoid(output[i]);
            }

            return output;
        }

        void computeGradient(unique_ptr<float[]> &X, unique_ptr<float[]> &y, unique_ptr<float[]> &y_pred, int samples){
            // get distance between y and y_pred
            unique_ptr<float[]> error = make_unique<float[]>(samples);
            for(int i = 0; i < samples; i++){
                error[i] = y_pred[i] - y[i];
            }

            // compute dW and dB
            // compute: dW = X.T * error
            // X (samples x feats) -> a (k, m)
            // error (samples x 1) -> b (k, n)
            // dW (feats x 1)        -> c (m, n)
            int m = input_features;
            int k = samples;
            int n = 1;
            float * a = X.get();
            float * b = error.get();
            float * c = dW.get();
            gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

            // compute dB = sum(error)
            dB = 0;
            for (int i = 0; i < samples; i++) {
                dB += error[i] / samples;
            }
        }

        void updateParams() {
            // weights = -lr * dW
            for (int i = 0; i < input_features; i++){
                weights[i] -= lr * dW[i];
            }

            // bias = -lr * dB
            bias -= lr * dB;
        }
    
    private:
        float lr = 0.001;
        int n_iters = 1000;
        int input_features = 0;
        unique_ptr<float[]> weights;
        unique_ptr<float[]> dW;
        float bias = 0;
        float dB = 0;
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
                datasetSize++;
            }
            assert(features.size() == ground_truth.size());
            trainSize = datasetSize * train_test_split;
            testSize = datasetSize - trainSize;
            featureSize = features.front().size();
        };

        int getTrainData(unique_ptr<float[]> &X, unique_ptr<float[]> &y) {
            X = make_unique<float[]>(trainSize*featureSize);
            y = make_unique<float[]>(trainSize);
            for (int i = 0; i < trainSize; i++){
                vector<float> feature = features[i];
                for(int j = 0; j < featureSize; j++){
                    X[i*featureSize + j] = feature[j];
                }
                y[i] = ground_truth[i];
            }
            return trainSize;
        };

        int getTestData(unique_ptr<float[]> &X, unique_ptr<float[]> &y) {
            X = make_unique<float[]>(testSize*featureSize);
            y = make_unique<float[]>(testSize);
            for (int i = 0; i < testSize; i++){
                vector<float> feature = features[i];
                for(int j = 0; j < featureSize; j++){
                    X[i*featureSize + j] = feature[j];
                }
                y[i] = ground_truth[i];
            }
            return testSize;
        };

        int getFeatureSize(){
            return featureSize;
        };
    
        int getTotalSize(){
            return datasetSize;
        };

    private:
        float train_test_split = 0.8;
        int trainSize = 0;
        int testSize = 0;
        int featureSize = 0;
        size_t datasetSize = 0;
        vector<vector<float>> features;
        vector<float> ground_truth;
};

int main(int argc, char** argv){
    /*
    TODO:
    1. Add shuffle to dataset
    */

   // Load dataset
    BreastCancerDataset dataset;
    int featSize = dataset.getFeatureSize();

    // Load train set
    unique_ptr<float[]> X_train;
    unique_ptr<float[]> y_train;
    int trainSamples = dataset.getTrainData(X_train, y_train);
    LogisticRegression LRModel{featSize, 0.001, 1000};

    // Model train
    LRModel.fit(X_train, y_train, trainSamples);

    // Load test set for evaluation
    unique_ptr<float[]> X_test;
    unique_ptr<float[]> y_test;
    int testSamples = dataset.getTestData(X_test, y_test);
    unique_ptr<float[]> y_pred = LRModel.predict(X_test, testSamples);

    float accuracy = LRModel.accuracy(y_test, y_pred, testSamples);
    LOG("Logistic Regression accuracy: " << setprecision(3) << accuracy << "%");

    return 0;
}