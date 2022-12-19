#include <iostream>
#include <memory>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <random>
#include <algorithm>
#include <unordered_map>

using namespace std;

#define LOG(x) cout << x << endl;

float rand_uniform(float min, float max) {
    if (max < min) {
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}

class BreastCancerDataset {
    public:
        BreastCancerDataset() {
            string filePath = "data/wdbc.data";

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

            normalize_data();
        };

        int getTrainData(unique_ptr<float[]> &X, unique_ptr<int[]> &y) {
            X = make_unique<float[]>(trainSize*featureSize);
            y = make_unique<int[]>(trainSize);
            for (int i = 0; i < trainSize; i++){
                vector<float> feature = features[i];
                for(int j = 0; j < featureSize; j++){
                    X[i*featureSize + j] = feature[j];
                }
                y[i] = ground_truth[i];
            }
            return trainSize;
        };

        int getTestData(unique_ptr<float[]> &X, unique_ptr<int[]> &y) {
            X = make_unique<float[]>(testSize*featureSize);
            y = make_unique<int[]>(testSize);
            // LOG(trainSize << " " << testSize << " " << datasetSize);
            for (int i = 0; i < testSize; i++){
                vector<float> feature = features[trainSize + i];
                for(int j = 0; j < featureSize; j++){
                    X[i*featureSize + j] = feature[j];
                }
                y[i] = ground_truth[trainSize + i];
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
        vector<int> ground_truth;
        unique_ptr<float[]> mean;
        unique_ptr<float[]> std;
        void normalize_data(){
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
            for (int i = 0; i < datasetSize; i++) {
                for (int j = 0; j < featureSize; j++) {
                    features[i][j] = (features[i][j] - mean[j]) / std[j];
                }
            }
            // ground truth is in range [0-1]
        };
};

class SVM {
    public:
        SVM(int input_feats): input_features{input_feats}{};

        SVM(int input_feats, float learning_rate, int n_iters, float lambda)
        : input_features{input_feats}, lr{learning_rate}, n_iters{n_iters}, lambda{lambda} {};

        void fit(unique_ptr<float[]> &X, unique_ptr<int[]> &y, int samples) {
            initParams();
            unique_ptr<int[]> convertedY = convert_label(y, samples); // convert y from [0,1] to [-1,1]
            for (int i = 0; i < n_iters; i++){
                for(int j = 0; j < samples; j++) {
                    //get constrain
                    int constraint = getConstraint(X.get() + j * input_features, convertedY[j]);
                    // LOG("Debug Constraint: " << constraint)

                    // compute gradient
                    computeGradient(constraint, X.get() + j * input_features, convertedY[j]);

                    // update gradient
                    updateParams();
                }
            }
        };

        unique_ptr<int[]> predict(unique_ptr<float[]> &input, int samples) {
            unique_ptr<int[]> y_pred = make_unique<int[]>(samples);
            for (int s = 0; s < samples; s++) {
                float h = 0;
                float * X_row = input.get() + s * input_features;
                for(int i = 0; i < input_features; i++) {
                        h += X_row[i] * weights[i];
                } 
                h += bias;
                y_pred[s] = h > 0 ? 1 : 0;
            }
            return y_pred;
        }

        float errorRate(unique_ptr<int[]> &y_gt, unique_ptr<int[]> &y_pred, int samples){
            return 100.f - accuracy(y_gt, y_pred, samples);
        }

        float accuracy(unique_ptr<int[]> &y_gt, unique_ptr<int[]> &y_pred, int samples){
            int true_count = 0;
            for (int i = 0; i < samples; i++) {
                if(y_gt[i] == y_pred[i]){
                    true_count++;
                }
            }
            return true_count * 100.f / samples;
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

        void updateParams(){
            for(int i = 0; i < input_features; i++) {
                weights[i] -= lr * dW[i];
            }
            bias -= lr*dB;
        }

        void computeGradient(int constraint, float * X_row, int y) {
            if(constraint) {
                for(int i = 0; i < input_features; i++) {
                    dW[i] = lambda * weights[i];
                    dB = 0;
                }
            }

            for(int i = 0; i < input_features; i++) {
                dW[i] = lambda * weights[i] - y * X_row[i];
                dB = -y;
            }
        }

        int getConstraint(float * X_row, int y) {
            float h = 0;
            for(int i = 0; i < input_features; i++) {
                    h += X_row[i] * weights[i];
            } 
            h += bias;
            return y * h >= 1;
        }

    private:
        unique_ptr<float[]> weights;
        unique_ptr<float[]> dW;
        float bias = 0;
        float dB = 0;
        float lr = 0.001;
        int n_iters = 1000;
        float lambda = 0.01;
        int input_features = 0;

        unique_ptr<int[]> convert_label(unique_ptr<int[]> &y, int samples) {
            auto converted = make_unique<int[]>(samples);
            for (int i = 0; i < samples; i++) {
                if(y[i] == 0) {
                    converted[i] = -1;
                } else {
                    converted[i] = y[i];
                }
            }
            return converted;
        }
};

int main(int argc, char** argv) {
    /*
    TODO:
    */

    // Load dataset
    BreastCancerDataset dataset;

    // Load train set
    unique_ptr<float[]> X_train;
    unique_ptr<int[]> y_train;
    int trainSamples = dataset.getTrainData(X_train, y_train);
    int featureSize = dataset.getFeatureSize();

    // Model train
    SVM model{featureSize};
    model.fit(X_train, y_train, trainSamples);

    // model predict
    unique_ptr<float[]> X_test;
    unique_ptr<int[]> y_test;
    int testSamples = dataset.getTestData(X_test, y_test);
    unique_ptr<int[]> y_pred = model.predict(X_test, testSamples);

    // Load test set for evaluation
    float accuracy = model.accuracy(y_test, y_pred, testSamples);

    LOG("SVM accuracy: " << setprecision(3) << accuracy << "%");
}