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
    void loadDataSet(QString path, QByteArray &byteArray, int offset);

public slots:
    void viewImage(QByteArray & ba, int imgIndex, QLabel * label);
    void setImageLabel(QByteArray & ba, int ind, QLabel *label);


private slots:
    void on_trainImagSeekSlider_actionTriggered(int action);

private:
    Ui::MainWindow *ui;
    NeuralNetwork *brain;
    QByteArray trainingImages;
    QByteArray trainingLabels;
    QByteArray testingImages;
    QByteArray testingLabels;
};

#endif // MAINWINDOW_H
