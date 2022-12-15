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

class IrisDataset {
    public:
        IrisDataset() {
            string filePath = "iris.data";

            ifstream file(filePath);

            if(!file.is_open()){
                LOG("Can not open dataset file");
            }

            string line = "";
            string value = "";
            string label = "";
            string currentLabel = "";
            int labelValue = 0;
            while(getline(file, line)) {
                stringstream ss(line);
                vector<float> feature;
                for (int i = 0; i < num_features; i++) {
                    getline(ss, value, ',');
                    feature.push_back(stof(value));
                }
                features.push_back(feature);
                getline(ss, label, ',');
                if(currentLabel == "") {
                    currentLabel = label;
                    labelValue = 0;
                } else {
                    // LOG(currentLabel << " " << label);
                    if(currentLabel != label){
                        labelValue++;
                        currentLabel = label;
                    }
                }
                ground_truth.push_back(labelValue);
                datasetSize++;
            }

            assert(features.size() == ground_truth.size());
            trainSize = datasetSize * train_test_split;
            testSize = datasetSize - trainSize;
            
            shuffle_data();
        }

        int getTrainData(unique_ptr<float[]> &X, unique_ptr<int[]> &y){
            X = make_unique<float[]>(trainSize*num_features);
            y = make_unique<int[]>(trainSize);
            for (int i = 0; i < trainSize; i++){
                vector<float> feature = features[i];
                for(int j = 0; j < num_features; j++){
                    X[i*num_features + j] = feature[j];
                }
                y[i] = ground_truth[i];
            }
            return trainSize;
        }

        int getTestData(unique_ptr<float[]> &X, unique_ptr<int[]> &y) {
            X = make_unique<float[]>(testSize*num_features);
            y = make_unique<int[]>(testSize);
            for (int i = 0; i < testSize; i++){
                vector<float> feature = features[i];
                for(int j = 0; j < num_features; j++){
                    X[i*num_features + j] = feature[j];
                }
                y[i] = ground_truth[i];
            }
            return testSize;
        };

        int getFeatureSize() {
            return num_features;
        }
    
    private:
        float train_test_split = 0.8;
        int trainSize = 0;
        int testSize = 0;
        int num_features = 4;
        size_t datasetSize = 0;
        vector<vector<float>> features;
        vector<int> ground_truth;
        unique_ptr<float[]> mean;
        unique_ptr<float[]> std;

        void shuffle_data() {
            shuffle(features.begin(), features.end(), std::default_random_engine(123));
            shuffle(ground_truth.begin(), ground_truth.end(), std::default_random_engine(123));
        }

        void normalize_data(){
            mean = make_unique<float[]>(num_features);
            std = make_unique<float[]>(num_features);
            // mean 
            for (int i = 0; i < trainSize; i++) {
                vector<float> feat = features[i];
                for (int j = 0; j < num_features; j++) {
                    mean[j] += feat[j];
                }
            }
            for (int i = 0; i < num_features; i++) {
                mean[i] /= trainSize;
                // LOG("mean: " << mean[i]);
            }

            // standard deviation
            for (int i = 0; i < trainSize; i++) {
                vector<float> feat = features[i];
                for (int j = 0; j < num_features; j++) {
                    std[j] += pow(feat[j] - mean[j], 2);
                }

            }

            for (int i = 0; i < num_features; i++) {
                std[i] = sqrt(std[i] / trainSize);
                // LOG("std: " << std[i]);
            }

            // normalize data
            for (int i = 0; i < datasetSize; i++) {
                for (int j = 0; j < num_features; j++) {
                    features[i][j] = (features[i][j] - mean[j]) / std[j];
                }
            }
            // ground truth is in range [0-1]
        };
};

void print(std::vector<int> const &v)
{
    for (int i: v) {
        std::cout << i << ' ';
    }
};

void test_shuffle(){
    std::vector<int> v = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::shuffle(v.begin(), v.end(),std::default_random_engine(123));
    print(v);
    std::cout << std::endl;

    std::vector<int> u = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    std::shuffle(u.begin(), u.end(),std::default_random_engine(123));
    print(u);
    std::cout << std::endl;

    for(int i = 0; i< 9; i++){
        assert(u[i] == (v[i] + 10));
    }
}

class KNN {
    public:
        KNN(int input_feats, int K): input_features{input_feats}, K{K}{

        };

        void fit(unique_ptr<float[]> &X, unique_ptr<int[]> &y, int samples) {
            this->X = X.get();
            this->y = y.get();
            num_samples = samples;
        };

        unique_ptr<int[]> predict(unique_ptr<float[]> &input, int samples) {
            unique_ptr<int[]> y_pred = make_unique<int[]>(samples);
            for(int i = 0; i < samples; i++) {
                float * row = input.get() + i * input_features;
                y_pred[i] = getSinglePrediction(row);
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


    private:
        float * X;
        int * y;
        int num_samples = 0;
        int input_features = 0;
        int K = 0;

        float getEuclideanDistance(float *x1, float *x2) {
            float sum_square_dist = 0.f;
            for(int i = 0; i < input_features; i++) {
                // cout << x1[i] << " " << x2[i] << " ";
                sum_square_dist += pow(x1[i] - x2[i], 2);
            }
            // cout << endl;
            return sqrt(sum_square_dist);
        }

        int getSinglePrediction(float *X_row) {
            unique_ptr<float[]> distances = make_unique<float[]>(num_samples);
            for (int i = 0; i < num_samples; i++ ){
                distances[i] = getEuclideanDistance(X_row, X + i*input_features);
                // LOG(distances[i])
            }

            // sort with index
            vector<pair<float, int>> distanceIndex;
            for(int i = 0; i < num_samples; i++) {
                distanceIndex.push_back(make_pair(distances[i], i));
            }
            sort(distanceIndex.begin(), distanceIndex.end());
            // LOG("debug distance source: ");
            // for (int i = 0; i < num_samples; i++ ){
            //     LOG(distanceIndex[i].first << " " << distanceIndex[i].second);
            // }
            unordered_map<int, int> KMap;
            for(int k = 0; k < K; k++) {
                int value = y[distanceIndex[k].second];
                if(KMap.find(value) == KMap.end()){
                    KMap[value] = 1;
                } else {
                    KMap[value] += 1;
                }
            }
            // LOG("debug KMap");
            // for (auto item : KMap) {
            //     LOG(item.first << " " << item.second);
            // }

            // return predict value
            int predictValue = 0;
            int KMaxCount = 0;
            for (auto item : KMap) {
                if(KMaxCount < item.second){
                    KMaxCount = item.second;
                    predictValue = item.first;
                }
            }

            return predictValue;
        }
};

int main(int argc, char** argv) {
    /*
    TODO:
    1. Load Iris dataset
    */
//    test_shuffle();

    // Load dataset
    IrisDataset dataset;

    // Load train set
    unique_ptr<float[]> X_train;
    unique_ptr<int[]> y_train;
    int trainSamples = dataset.getTrainData(X_train, y_train);
    int featureSize = dataset.getFeatureSize();

    // Model train
    int K = 3;
    KNN model{featureSize, K};
    model.fit(X_train, y_train, trainSamples);

    // model predict
    unique_ptr<float[]> X_test;
    unique_ptr<int[]> y_test;
    int testSamples = dataset.getTestData(X_test, y_test);
    unique_ptr<int[]> y_pred = model.predict(X_test, testSamples);

    // Load test set for evaluation
    float accuracy = model.accuracy(y_test, y_pred, testSamples);

    LOG("KNN accuracy: " << setprecision(3) << accuracy << "%");
}