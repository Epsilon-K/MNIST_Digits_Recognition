#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    loadStyle();
    showMaximized();
    srand(time(0));


    QTimer::singleShot(200,this,SLOT(loadData()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::loadDataSet(QString path, QVector<Matrix *> &vm, int offset)
{
    QFile file(path);
      if (!file.open(QIODevice::ReadOnly)){
          qDebug() << "Can't read the file" << path;
          return;
      }

    QByteArray *ba = new QByteArray(file.readAll());
    ba->remove(0,offset);
    if(offset == 16){ // loading an images file
        for(int i = 0; i < ba->size()/784; i++){    // number of images
            QVector<double> v;
            for(int j = 0; j < 784; j++){
                v.append(uchar(ba->at(i*784 + j)));
            }
            Matrix *m = new Matrix(v);
            vm.append(m);
            load += 784;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54900000 * 100));
        }
    }else{  // loading the labels
        for(int i = 0; i < ba->size(); i++){
            uchar uc = uchar(ba->at(i));
            QVector<double> v;
            for(int j = 0; j < 10; j++){
                v.append(uchar(j) == uc ? 1 : 0);
            }
            Matrix *m = new Matrix(v);
            vm.append(m);
            load ++;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54900000 * 100));
        }
    }
}


void MainWindow::viewImage(QVector<Matrix *> vm, int imgIndex, QLabel *label)
{
    // create uchar array
    uchar data[784];

    for(int i = 0; i < 784; i++){
        data[i] = uchar(vm[imgIndex]->data.at(i).at(0));
    }

    // create QImage & QPixmap
    QImage img(data, 28, 28, QImage::Format_Grayscale8);
    QPixmap pix = QPixmap::fromImage(img);

    // send to view on the Label
    label->setPixmap(pix.scaled(28*5,28*5));
}

void MainWindow::setImageLabel(QVector<Matrix *> vm, int ind, QLabel *label)
{
    QString str = "image[" + QString::number(ind) + "] Label : ";
    str += QString::number(uchar(vm[ind]->data.indexOf({1})));
    label->setText(str);
}

void MainWindow::loadStyle()
{
    QFile file(":/style/darkorange.stylesheet");
      if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
          qDebug() << "Can't read the file :/style/darkorange.stylesheet";
          return;
      }
    setStyleSheet(QString(file.readAll()));
}

void MainWindow::on_trainImagSeekSlider_valueChanged(int value)
{
    viewImage(trainingImages, value, ui->traingingImagesLabel);
    setImageLabel(trainingLabels, value, ui->trainImgLabel);
}

void MainWindow::on_testImgSeekSlider_valueChanged(int value)
{
    viewImage(testingImages, value, ui->testingImagesLabel);
    setImageLabel(testingLabels, value, ui->testImgLabel);
}

void MainWindow::loadData()
{
    // load dataset
    loadDataSet(":/dataset/train-labels.idx1-ubyte", trainingLabels, 8);
    loadDataSet(":/dataset/train-images.idx3-ubyte", trainingImages, 16);
    loadDataSet(":/dataset/t10k-labels.idx1-ubyte", testingLabels, 8);
    loadDataSet(":/dataset/t10k-images.idx3-ubyte", testingImages, 16);

    // view 1st Training Image & 1st Testing Image
    on_trainImagSeekSlider_valueChanged(0);
    on_testImgSeekSlider_valueChanged(0);
}

