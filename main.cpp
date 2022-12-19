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
            string filePath = "data/iris.data";

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
            
            // normalize_data();
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
                vector<float> feature = features[trainSize + i];
                for(int j = 0; j < num_features; j++){
                    X[i*num_features + j] = feature[j];
                }
                y[i] = ground_truth[trainSize + i];
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
            // ground truth already in range [0-1]
        };
};


class NaiveBayes {
    public:
        NaiveBayes(int n_classes, int input_feats): input_features{input_feats}, n_classes{n_classes}{
            mean = make_unique<float[]>(n_classes * input_features); // mean of each feature for each class
            variance = make_unique<float[]>(n_classes * input_features); // variance of each feature for each class
            priors = make_unique<float[]>(n_classes);
        };

        void fit(unique_ptr<float[]> &X, unique_ptr<int[]> &y, int samples) {
            // compute all classes features mean and variance

            // sum of each feature in each class
            unique_ptr<float[]> cls_count = make_unique<float[]>(n_classes);
            unique_ptr<float[]> mean_of_square = make_unique<float[]>(n_classes * input_features);
            for(int i = 0; i < samples; i++) {
                int cls= y[i];
                cls_count[cls]++;
                for(int f = 0; f < input_features; f++){
                    mean[cls*input_features + f] += X[i*input_features + f];
                    mean_of_square[cls*input_features+f] += pow(X[i*input_features + f], 2);
                }
            }
            
            // mean of every features for each class
            for(int c = 0; c < n_classes; c++){
                for(int f = 0; f < input_features; f++) {
                    mean[c*input_features + f] /= cls_count[c];
                    mean_of_square[c*input_features+f] /= cls_count[c];
                }
            }

            // LOG("Debug mean: ");
            // for(int c = 0; c < n_classes; c++){
            //     LOG("class " << c);
            //     for(int f = 0; f < input_features; f++){
            //         LOG(mean[c*input_features + f]);
            //     }
            // }

            // variance
            for(int c = 0; c < n_classes; c++){
                for(int f = 0; f < input_features; f++) {
                    variance[c*input_features + f] = mean_of_square[c*input_features + f] - pow(mean[c*input_features+f], 2);
                }
            }

            // LOG("Debug variance: ");
            // for(int c = 0; c < n_classes; c++){
            //     LOG("class " << c);
            //     for(int f = 0; f < input_features; f++){
            //         LOG(variance[c*input_features + f]);
            //     }
            // }

            // priors
            for(int c = 0; c < n_classes; c++) {
                priors[c] = cls_count[c] / samples;
            }

            // LOG("Debug prior: ");
            // for(int c = 0; c < n_classes; c++) {
            //     LOG("class " << c);
            //     LOG(priors[c]);
            // }
        };

        unique_ptr<int[]> predict(unique_ptr<float[]> &input, int samples) {
            unique_ptr<int[]> y_pred = make_unique<int[]>(samples);
            for(int i = 0; i < samples; i++) {
                y_pred[i] = getClassPrediction(input.get() + i * input_features);
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
        unique_ptr<float[]> mean; 
        unique_ptr<float[]> variance;
        unique_ptr<float[]> priors;
        float lr = 0.001;
        int n_iters = 1000;
        int input_features = 0;
        int n_classes = 0;

        int getClassPrediction(float * X_row) {
            unique_ptr<float[]> posteriors = make_unique<float[]>(n_classes);
            for(int c = 0; c < n_classes; c++){
                float * cls_mean = mean.get() + c*input_features;
                float * cls_variance = variance.get() + c*input_features;
                float cls_prior = log(priors[c]);

                unique_ptr<float[]> probabilities = gaussianDensity(X_row, cls_mean, cls_variance);
                for(int i = 0; i < input_features; i++){
                    posteriors[c] += log(probabilities[i] + 0.000001);
                }
                posteriors[c] += cls_prior;
            }

            float max_posterior = posteriors[0];
            int predict_class = 0;
            for(int c = 0; c < n_classes; c++) {
                if(max_posterior < posteriors[c]) {
                    max_posterior = posteriors[c];
                    predict_class = c;
                }
            }

            return predict_class;
        }

        unique_ptr<float[]> gaussianDensity(float * X_row, float * cls_mean,  float * cls_var) {
            unique_ptr<float[]> proba = make_unique<float[]>(input_features);
            for (int i = 0; i < input_features; i++){
                float constant = 1 / sqrt(cls_var[i] * 2 * M_PI);
                proba[i] = constant * exp(-0.5f * (pow(X_row[i] - cls_mean[i], 2) / cls_var[i]));
            }

            return proba;
        }
};

int main(int argc, char** argv) {
    /*
    TODO:
    */

    // Load dataset
    IrisDataset dataset;

    // // Load train set
    unique_ptr<float[]> X_train;
    unique_ptr<int[]> y_train;
    int trainSamples = dataset.getTrainData(X_train, y_train);
    int featureSize = dataset.getFeatureSize();

    // // Model train
    NaiveBayes model{3, featureSize};
    model.fit(X_train, y_train, trainSamples);

    // // model predict
    unique_ptr<float[]> X_test;
    unique_ptr<int[]> y_test;
    int testSamples = dataset.getTestData(X_test, y_test);
    unique_ptr<int[]> y_pred = model.predict(X_test, testSamples);

    // // Load test set for evaluation
    float accuracy = model.accuracy(y_test, y_pred, testSamples);

    LOG("Naive Bayes accuracy: " << setprecision(3) << accuracy << "%");
}