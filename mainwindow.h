#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <NeuralNetwork.h>
#include <QFile>
#include <QLabel>
#include <QImage>
#include <QPixmap>
#include <QMessageBox>
#include <QRegularExpression>
#include <QFileDialog>
#include <QTime>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void loadDataSet(QString path, QVector<QVector<uchar>> &data, int offset);

public slots:
    void viewImage(QVector<QVector<uchar> > &data, int imgIndex, QLabel *label);
    void setImageLabel(QVector<QVector<uchar>> &data, int ind, QLabel *label);
    void loadStyle();
    void loadData();
    QString setNNFullName();
    void feedImage();
    void train();
    QString getRandomName(int len);
    void save();
    void proof();


private slots:
    void on_trainImagSeekSlider_valueChanged(int value);

    void on_testImgSeekSlider_valueChanged(int value);

    void on_testNNBtn_clicked();

    void on_saveModelBtn_clicked();

    void on_startTrainingBtn_clicked();

    void on_createModelBtn_clicked();

    void on_loadModelSpinBox_clicked();

    void deleteDataLoadingBar();

    void on_checkBox_toggled(bool checked);

private:
    Ui::MainWindow *ui;
    NeuralNetwork *brain;
    QVector<QVector<uchar>> trainingImages;
    QVector<QVector<uchar>> trainingLabels;
    QVector<QVector<uchar>> testingImages;
    QVector<QVector<uchar>> testingLabels;
    double load = 0;

    int testIndex = 0;
    int correctTests = 0;

    int epochIndex = 0;
    int batchIndex = 0;
    int trainIndex = 0;

    bool realTime;
    QTime tiem;
};

#endif // MAINWINDOW_H
