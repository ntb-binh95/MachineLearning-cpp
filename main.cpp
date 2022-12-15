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

class HousePriceDataset {
    public:
        HousePriceDataset() {
            string filePath = "house_price_data.txt";
            ifstream file(filePath);

            if(!file.is_open()){
                LOG("Cannot open dataset file");
            }

            string line = "";
            string value = "";
            string price = "";
            while(getline(file, line)) {
                stringstream ss{line};
                vector<float> feature;
                for (int i = 0; i < 2; i++) {
                    getline(ss, value, ',');
                    feature.push_back(stof(value));
                }
                features.push_back(feature);
                getline(ss, price, ',');
                ground_truth.push_back(stof(price));
                datasetSize++;
            }
            assert(features.size() == ground_truth.size());
            trainSize = datasetSize * train_test_split;
            testSize = datasetSize - trainSize;
            featureSize = features.back().size();
        }

        int getTrainData(unique_ptr<float[]> &X, unique_ptr<float[]> &y) {
            X = make_unique<float[]>(trainSize*featureSize);
            y = make_unique<float[]>(trainSize);
            mean = make_unique<float[]>(featureSize);
            std = make_unique<float[]>(featureSize);
            // mean 
            for (int i = 0; i < trainSize; i++) {
                vector<float> feat = features[i];
                for (int j = 0; j < featureSize; j++) {
                    mean[j] += feat[j];
                }
            }
            for (int i = 0; i < featureSize; i++) {
                mean[i] /= trainSize;
                // LOG("mean: " << mean[i]);
            }

            // standard deviation
            for (int i = 0; i < trainSize; i++) {
                vector<float> feat = features[i];
                for (int j = 0; j < featureSize; j++) {
                    std[j] += pow(feat[j] - mean[j], 2);
                }

            }

            for (int i = 0; i < featureSize; i++) {
                std[i] = sqrt(std[i] / trainSize);
                // LOG("std: " << std[i]);
            }

            // normalize data
            for (int i = 0; i < trainSize; i++) {
                vector<float> feat = features[i];
                for (int j = 0; j < featureSize; j++) {
                    X[i*featureSize + j] = (feat[j] - mean[j]) / std[j];
                    // LOG(X[i*featureSize + j]);
                }
            }
            for(int i =0; i < trainSize; i++) {
                y[i] = ground_truth[i];
                // LOG(y[i]);
            }

            return trainSize;
        }

        void preprocess(unique_ptr<float[]> &X, int samples){
            for(int i = 0; i < samples; i++) {
                for(int j = 0; j < featureSize; j++) {
                    X[i*featureSize + j] = (X[i*featureSize + j] - mean[j]) / std[j];
                }
            }
        }

    private:
        float train_test_split = 1.0f;
        vector<vector<float>> features;
        vector<float> ground_truth;
        unique_ptr<float[]> mean;
        unique_ptr<float[]> std;
        int datasetSize = 0;
        int trainSize = 0;
        int testSize = 0;
        int featureSize = 0;
};

class LinearRegression {
    public:
        LinearRegression(int input_feats): input_features{input_feats} {};
        LinearRegression(int input_feats, float learning_rate, int n_iters): 
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
            auto predict = getPrediction(X, samples);
            return predict;
        }

        float score(unique_ptr<float[]> &y_gt, unique_ptr<float[]> &y_pred, int samples){
            float y_mean = 0.f;
            for(int i = 0; i < samples; i++){
                y_mean += y_gt[i];
            }
            y_mean /= samples;
                
            float upper = 0.f;
            float lower = 0.f;
            for(int i = 0; i < samples; i++){
                upper += pow(y_gt[i] - y_pred[i], 2);
                lower += pow(y_gt[i] - y_mean, 2);
            }
            return 1 - upper/lower;
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
            // weights (feats x 1)       | b(k,n)
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
            // dW (feats x 1)      -> c (m, n)
            int m = input_features;
            int k = samples;
            int n = 1;
            float * a = X.get();
            float * b = error.get();
            float * c = dW.get();
            gemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);

            // for (int i = 0; i < input_features; i++){
            //     LOG("DW " << X[i]);
            // }
            // compute dB = sum(error)
            dB = 0;
            for (int i = 0; i < samples; i++) {
                dB += error[i] / samples;
            }
            // LOG(dB);
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

int main(int argc, char** argv){
    /*
    TODO:
    1. Add shuffle to dataset
    */

    HousePriceDataset dataset;
    unique_ptr<float[]> X_train;
    unique_ptr<float[]> y_train;
    int trainSize = dataset.getTrainData(X_train, y_train);

    int featureSize = 2;
    LinearRegression model{featureSize, 0.01, 1000};

    model.fit(X_train, y_train, trainSize);

    unique_ptr<float[]> X_test = make_unique<float[]>(featureSize);
    X_test[0] = 4478;
    X_test[1] = 5;
    // dataset.preprocess(X_test, 1);
    // LOG(X_test[0] << " " << X_test[1]);
    auto y_test = model.predict(X_train, trainSize);

    float score = model.score(y_train, y_test, trainSize);
    LOG("Model score: " << score);

    return 0;
}