#ifndef NN_H
#define NN_H
#include <matrix.h>

class NeuralNetwork : public QObject
{
    Q_OBJECT
public:
    QString name;
    QString path;
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


    NeuralNetwork(QString _path, QString _name, QVector<int> _structure, int _batchSize = 100, int _epochs = 1,
                  double _learningRate = 0.05) {
        path = _path;
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

    bool isSameStructure(NeuralNetwork * nn){
        if(structure.size() == nn->structure.size()){
            for(int i = 0; i < structure.size(); i++){
                if(structure[i] != nn->structure[i]) return false;
            }
            return true;
        }
        return false;
    }

    void copyWnB(NeuralNetwork * nn){
        weights = nn->weights;
        biases = nn->biases;
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
        //weights[weights.size()-1]->add(delta);
        //biases[biases.size()-1]->add(gradient);

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
            //weights[i]->add(lDelta);
            //biases[i]->add(lGradient);

            gradients[i]->add(lGradient);
            deltas[i]->add(lDelta);

            delete lGradient;
            delete tr;
            delete lDelta;
        }

        return outputs;
    }

    template< typename T >
    void shuffleVector(QVector<T> &vec1, QVector<T> &vec2){
        for(int i = 0; i < vec1.size(); i++){
            int r = rand()%vec1.size();

            T t = vec1.at(i);
            vec1[i] = vec1.at(r);
            vec1[r] = t;

            T t2 = vec2.at(i);
            vec2[i] = vec2.at(r);
            vec2[r] = t2;
        }
    }

    QString toString(){
        QString str;
        QTextStream stream(&str);
        stream << path << "\n";
        stream << name << " {";
        for(int i = 0 ; i < structure.size(); i++){
            stream << structure[i];
            if(i < structure.size()-1) stream << ", ";
        }
        stream << "} Batch(" << batchSize << ") Epoch(" << epochs
               << ") LR(" << learningRate << ")\n";
        for(int i = 0; i < weights.size(); i++){
            stream << weights[i]->toString();
        }
        for(int i = 0; i < biases.size(); i++){
            stream << biases[i]->toString();
        }
        return str;
    }

    static NeuralNetwork* fromString(QString str){
        // extract NN info
        QString path = rm(str, 0, str.indexOf('\n')); str.remove(0,1);
        QString name = rm(str, 0, str.indexOf(' ')); str.remove(0,2);
        QStringList structNums = rm(str, 0, str.indexOf('}')).split(", ",QString::SkipEmptyParts);
        str.remove(0,8);
        QVector<int> structV;
        for(int i = 0; i < structNums.size(); i++) structV.append(structNums[i].toInt());
        int batch = rm(str, 0, str.indexOf(')')).toInt(); str.remove(0, 8);
        int epoch = rm(str, 0, str.indexOf(')')).toInt(); str.remove(0, 5);
        double lr = rm(str, 0, str.indexOf(')')).toDouble(); str.remove(0, 2);

        NeuralNetwork * nn = new NeuralNetwork(path, name, structV, batch, epoch, lr);

        QStringList WnBR = str.split('\n', QString::SkipEmptyParts);
        int ind = 0;
        // extract weights
        for(int w = 0; w < nn->weights.size(); w++){
            for(int wr = 0; wr < nn->weights[w]->rows(); wr++){
                QStringList wrv = WnBR[ind].split(" ");
                for(int wrj = 0; wrj < nn->weights[w]->cols(); wrj++){
                    nn->weights[w]->data[wr][wrj] = wrv[wrj].toDouble();
                }
                ind++;
            }
        }
        // extract Biases
        for(int b = 0; b < nn->biases.size(); b++){
            for(int br = 0; br < nn->biases[b]->rows(); br++){
                QStringList brv = WnBR[ind].split(" ");
                for(int brj = 0; brj < nn->biases[b]->cols(); brj++){
                    nn->biases[b]->data[br][brj] = brv[brj].toDouble();
                }
                ind++;
            }
        }
        return nn;
    }

    static QString rm(QString &str, int ind, int len){
        QString s = str.left(len);
        str.remove(ind, len);
        return s;
    }
};

#endif // NN_H


















