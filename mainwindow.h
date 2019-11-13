#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <NeuralNetwork.h>
#include <QFile>
#include <QLabel>
#include <QImage>
#include <QPixmap>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void loadDataSet(QString path, QVector<Matrix*> &vm, int offset);

public slots:
    void viewImage(QVector<Matrix*> vm, int imgIndex, QLabel *label);
    void setImageLabel(QVector<Matrix*> ba, int ind, QLabel *label);


private slots:
    void on_trainImagSeekSlider_valueChanged(int value);

    void on_testImgSeekSlider_valueChanged(int value);

private:
    Ui::MainWindow *ui;
    NeuralNetwork *brain;
    QVector<Matrix*> trainingImages;
    QVector<Matrix*> trainingLabels;
    QVector<Matrix*> testingImages;
    QVector<Matrix*> testingLabels;
};

#endif // MAINWINDOW_H
