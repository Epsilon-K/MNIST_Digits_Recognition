#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    srand(time(0));

    // load dataset
    loadDataSet(":/dataset/train-labels.idx1-ubyte", trainingLabels, 8);
    loadDataSet(":/dataset/t10k-labels.idx1-ubyte", testingLabels, 8);
    loadDataSet(":/dataset/train-images.idx3-ubyte", trainingImages, 16);
    loadDataSet(":/dataset/t10k-images.idx3-ubyte", testingImages, 16);

    // view 1st Training Image & 1st Testing Image
    on_trainImagSeekSlider_actionTriggered(0);
    viewImage(testingImages, 0, ui->testingImagesLabel);
    setImageLabel(testingLabels, 0, ui->testImgLabel);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::loadDataSet(QString path, QByteArray &byteArray, int offset)
{
    QFile file(path);
      if (!file.open(QIODevice::ReadOnly)){
          qDebug() << "Can't read the file" << path;
          return;
      }

        byteArray = file.readAll();
        byteArray.remove(0,offset);
}

void MainWindow::viewImage(QByteArray &ba, int imgIndex, QLabel *label)
{
    // create uchar array
    uchar data[784];
    int startInd = imgIndex * 784;
    int finishInd = startInd + 784;

    for(int i = startInd; i < finishInd; i++){
        data[i] = uchar(ba.data()[i]);
    }

    // create QImage & QPixmap
    QImage img(data, 28, 28, QImage::Format_Grayscale8);
    QPixmap pix = QPixmap::fromImage(img);

    // send to view on the Label
    label->setPixmap(pix.scaled(28*5,28*5));
}

void MainWindow::setImageLabel(QByteArray &ba, int ind, QLabel *label)
{
    QString str = "image[" + QString::number(ind) + "] Label : ";
    str += QString::number(uchar(ba.at(ind)));
    label->setText(str);
}

void MainWindow::on_trainImagSeekSlider_actionTriggered(int action)
{
    qDebug() << "Gonna Seek to " << QString::number(action);
    viewImage(trainingImages, action, ui->traingingImagesLabel);
    qDebug() << "Viewed Image!";
    setImageLabel(trainingLabels, action, ui->trainImgLabel);
    qDebug() << "Set the Label";
}
