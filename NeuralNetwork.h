#ifndef NN_H
#define NN_H
#include <matrix.h>

class NeuralNetwork
{
public:
    QVector<int> structure;
    QVector<Matrix> weights;
    QVector<Matrix> biases;
    QVector<Matrix> layers;
    QVector<Matrix> errors;

    double learningRate = 0.05;

    NeuralNetwork(QVector<int> _structure) {
        structure = _structure;
        // create the layers
        for(int i = 0; i < structure.size(); i++){
            Matrix m(structure[i], 1);
            layers.append(m);
        }
        // create weights, biases And errors
        for(int i = 1; i < structure.size(); i++){
            Matrix weight(structure[i], structure[i-1]);
            weight.randomize();
            weights.append(weight);

            Matrix bias(structure[i], 1);
            bias.add(1); // or just zeros???
            biases.append(bias);

            Matrix err(structure[i], 1);
            errors.append(err);
        }

    }

    Matrix feedForward(Matrix inputs){
        layers[0].copy(inputs);
        for(int i = 0; i < weights.size(); i++){
            layers[i+1].copy(Matrix::dotProduct(weights[i], layers[i]));
            layers[i+1].add(biases[i]);
            // activation
            layers[i+1].map(Matrix::sigmoid);
        }
        return layers[layers.size()-1];
    }

    Matrix backPropagation(Matrix inputs, Matrix targets){
        Matrix outputs = this->feedForward(inputs);

        // Calculate error for output layer
        errors[errors.size()-1] = Matrix::subtract(targets, outputs);
        Matrix gradient(outputs);

        //Calculate Gradient
        gradient.map(Matrix::dSigmoid);
        gradient.multiply(errors[errors.size()-1]);
        gradient.multiply(learningRate);

        //Calculate Delta
        Matrix lt(Matrix::transpose(layers[layers.size()-2]));
        Matrix delta(Matrix::dotProduct(gradient,lt));

        //Apply the change to weight
        weights[weights.size()-1].add(delta);
        biases[biases.size()-1].add(gradient);

        // Calculate Errors
        for(int i = errors.size()-2; i >= 0; i--){
            //err
            Matrix wt(Matrix::transpose(weights[i+1]));
            errors[i].copy(Matrix::dotProduct(wt,errors[i+1]));
            //Gradient
            Matrix lGradient(layers[i+1]);
            lGradient.map(Matrix::dSigmoid);
            lGradient.multiply(errors[i]);
            lGradient.multiply(learningRate);
            //delta
            Matrix tr(Matrix::transpose(layers[i]));
            Matrix lDelta(Matrix::dotProduct(lGradient, tr));

            // Apply delta to the weights
            weights[i].add(lDelta);
            biases[i].add(lGradient);
        }
        return outputs;
    }
};

#endif // NN_H


















