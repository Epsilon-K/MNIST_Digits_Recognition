#ifndef NN_H
#define NN_H
#include <matrix.h>

class NeuralNetwork : public QObject
{
    Q_OBJECT
public:
    QString name;
    QVector<int> structure;
    QVector<Matrix*> weights;
    QVector<Matrix*> biases;
    QVector<Matrix*> layers;
    QVector<Matrix*> errors;

    QVector<Matrix*> deltas;
    QVector<Matrix*> gradients;

    double learningRate = 0.05;
    int batchSize = 100;
    int epochs = 3;


    NeuralNetwork(QString _name, QVector<int> _structure, int _batchSize = 100, int _epochs = 1,
                  double _learningRate = 0.05) {
        name = _name;
        structure = _structure;
        batchSize = _batchSize <= 0 ? 1 : _batchSize;
        epochs = _epochs;  learningRate = _learningRate;

        // create the layers
        for(int i = 0; i < structure.size(); i++){
            Matrix *m = new Matrix(structure[i], 1);
            layers.append(m);
        }
        // create weights, biases, errors, deltas, and gradients
        for(int i = 1; i < structure.size(); i++){
            Matrix *weight = new Matrix(structure[i], structure[i-1]);
            weight->randomize();
            weights.append(weight);

            Matrix *bias = new Matrix(structure[i], 1);
            biases.append(bias);

            Matrix *err = new Matrix(structure[i], 1);
            errors.append(err);


            Matrix *gradient = new Matrix(structure[i], 1);
            gradients.append(gradient);

            Matrix *delta = new Matrix(structure[i], structure[i-1]);
            deltas.append(delta);
        }
    }

    Matrix* feedForward(Matrix *inputs){
        layers[0]->copy(inputs);
        for(int i = 0; i < weights.size(); i++){
            Matrix *dp = Matrix::dotProduct(weights[i], layers[i]);
            layers[i+1]->copy(dp); delete dp;
            layers[i+1]->add(biases[i]);
            // activation
            layers[i+1]->map(Matrix::sigmoid);
        }
        return layers[layers.size()-1];
    }

    // --------------------------------------------------------------------
    Matrix* backPropagation(Matrix *inputs, Matrix *targets){
        Matrix *outputs = this->feedForward(inputs);

        // Calculate error for output layer
        delete errors[errors.size()-1];
        errors[errors.size()-1] = Matrix::subtract(targets, outputs);
        Matrix *gradient = new Matrix(outputs);

        //Calculate Gradient
        gradient->map(Matrix::dSigmoid);
        gradient->multiply(errors[errors.size()-1]);
        gradient->multiply(learningRate);

        //Calculate Delta
        Matrix *lt = Matrix::transpose(layers[layers.size()-2]);
        Matrix *delta = Matrix::dotProduct(gradient,lt);

        //adjust!!!
        weights[weights.size()-1]->add(delta);
        biases[biases.size()-1]->add(gradient);

        deltas[deltas.size()-1]->add(delta);
        gradients[gradients.size()-1]->add(gradient);

        delete gradient;
        delete lt;
        delete delta;

        // Calculate Errors
        for(int i = errors.size()-2; i >= 0; i--){
            //err
            Matrix *wt = Matrix::transpose(weights[i+1]);
            Matrix *dp = Matrix::dotProduct(wt,errors[i+1]); delete wt;
            errors[i]->copy(dp); delete dp;
            //Gradient
            Matrix *lGradient = new Matrix(layers[i+1]);
            lGradient->map(Matrix::dSigmoid);
            lGradient->multiply(errors[i]);
            lGradient->multiply(learningRate);
            //delta
            Matrix *tr = Matrix::transpose(layers[i]);
            Matrix *lDelta = Matrix::dotProduct(lGradient, tr);

            // Adjust-NOT
            weights[i]->add(lDelta);
            biases[i]->add(lGradient);

            gradients[i]->add(lGradient);
            deltas[i]->add(lDelta);

            delete lGradient;
            delete tr;
            delete lDelta;
        }

        return outputs;
    }

    /*void training(QVector<Matrix*> inputs, QVector<Matrix*> targets){
        //01 Run the epoch loop
        for(int epoch = 0; epoch < epochs; epoch++){
            //02 Random Shuffle the data
            shuffleVector(inputs, targets);

            //03 Batch Loop
            int dataIndex = 0;
            while(dataIndex < inputs.size()){
                for(int batch = 0; (batch < batchSize && dataIndex < inputs.size())
                    ; batch++){
                    //04 backpropgation
                    backPropagation(inputs[dataIndex], targets[dataIndex]);
                    dataIndex++;
                }

                for(int i = 0; i < weights.size(); i++){
                    //05 divide the already summed gradients
                    //   and deltas by batchSize to get Average
                    gradients[i]->divide(batchSize);
                    deltas[i]->divide(batchSize);

                    //06 Adjust weights and biases
                    biases[i]->add(gradients[i]);
                    weights[i]->add(deltas[i]);
                }
            }
            //07 report cost function after epoch[epoch]
            // TODO!!!
        }
    }
    */

    void shuffleVector(QVector<Matrix *> &vec1, QVector<Matrix *> &vec2){
        for(int i = 0; i < vec1.size(); i++){
            int r = rand()%vec1.size();

            Matrix * t = vec1.at(i);
            vec1[i] = vec1.at(r);
            vec1[r] = t;

            Matrix * t2 = vec2.at(i);
            vec2[i] = vec2.at(r);
            vec2[r] = t2;
        }
    }
};

#endif // NN_H


















