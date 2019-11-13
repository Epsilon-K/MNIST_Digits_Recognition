#ifndef MATRIX_H
#define MATRIX_H
#include <QObject>
#include <QVector>
#include <QString>
#include <QDebug>
#include <math.h>

class Matrix{
public:
    QVector<QVector<double>> data;
    int rows(){
        return data.size();
    }
    int cols(){
        return data.at(0).size();
    }


    QVector<double> operator[](int i){
        return data[i];
    }


    Matrix(int rows, int cols){
        for(int i = 0; i < rows; i++){
            QVector<double> v;
            for(int j = 0; j < cols; j++){
                v.append(0);
            }
            data.append(v);
        }
    }

    Matrix(Matrix *mat){
        data = mat->data;
    }

    /**
     * @brief copys mat.data into this->data
     * @param mat
     */
    void copy(Matrix *mat){
        data = mat->data;
    }

    Matrix (QVector<double> v){
        for(int i = 0; i < v.size(); i++){
            data.append(QVector<double>{v[i]});
        }
    }



    void add(double n){
        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] += n;
            }
        }
    }

    void add(Matrix *mat){
        if(mat->rows() != rows() || mat->cols() != cols()){
            qDebug() << "MATRIX ERROR : Matrix Mismatch in Function add(Matrix mat)";
            return;
        }

        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] += mat->data.at(i).at(j);
            }
        }
    }

    static Matrix* subtract(Matrix *a, Matrix *b){
        if(a->rows() != b->rows() || a->cols() != b->cols()){
            qDebug() << "MATRIX ERROR : Matrix Mismatch in Function subtract(Matrix mat)";
            return a;
        }

        Matrix *mat = new Matrix(a->rows(), a->cols());
        for(int i = 0; i < a->rows(); i++){
            for(int j = 0; j < a->cols(); j++){
                mat->data[i][j] = a->data.at(i).at(j) - b->data.at(i).at(j);
            }
        }
        return mat;
    }

    void divide(double n){
        if(n < 1){
            qDebug() << "Matrix Error : n is less than 1, in Function divide(double n)";
            return ;
        }
        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] /= n;
            }
        }
    }

    void multiply(double n){
        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] *= n;
            }
        }
    }

    void multiply(Matrix mat){
        if(mat.rows() != rows() || mat.cols() != cols()){
            qDebug() << "MATRIX ERROR : Matrix Mismatch in Function multiply(Matrix mat)";
            return;
        }

        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] *= mat.data[i][j];
            }
        }
    }

    static Matrix *dotProduct(Matrix *a, Matrix *b){
        if(a->cols() != b->rows()){
            qDebug() << "MATRIX ERROR in dotProduct(Matrix mat) : A.cols["
                        +QString::number(a->cols()) + "] != B.rows[" +
                        QString::number(b->rows()) + "]";
            return a;
        }
        Matrix *nmat = new Matrix(a->rows(), b->cols());
        for(int i = 0; i < nmat->rows(); i++){
            for (int j = 0; j < nmat->cols(); j++) {
                for(int k = 0; k < a->cols(); k++){
                    nmat->data[i][j] += (a->data.at(i).at(k) * b->data.at(k).at(j));
                }
            }
        }
        return nmat;
    }

    static Matrix* transpose(Matrix *a){
        Matrix *mat = new Matrix(a->cols(), a->rows());
        for(int i = 0; i < a->rows(); i++){
            for(int j = 0; j < a->cols(); j++){
                mat->data[j][i] = a->data.at(i).at(j);
            }
        }
        return mat;
    }

    void map(double (*fn)(double)){
        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] = fn(data[i][j]);
            }
        }
    }

    static double doubleIt(double x){
        return x*2;
    }

    static double sigmoid(double x){
        return 1/(1 + std::exp(-x));
    }

    static double dSigmoid(double y){
        //return sigmoid(x) * (1 - sigmoid(x));
        return y * (1 - y);
    }

    static double rnd(){
        double d = double(rand())/RAND_MAX;
        d *= 2; d -= 1;
        return d;
    }

    void randomize(){
        for(int i = 0; i < rows(); i++){
            for(int j = 0; j < cols(); j++){
                data[i][j] = rnd();
            }
        }
    }

    QString toString(){
        QString str = "[";
        for(int i = 0; i < rows(); i++){
            if(i > 0) str += " ";
            str += "[";
            for(int j = 0; j < cols(); j++){
                str += QString::number(data[i][j]);
                if(j < cols()-1) str += ", ";
            }
            str += "]";
            if(i < rows()-1) str += "\n";
        }
        str+= "]";
        return str;
    }

};

#endif // MATRIX_H
