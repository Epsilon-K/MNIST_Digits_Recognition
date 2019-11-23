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

    ui->modelNameLineEdit->setText(getRandomName(3));

    QTimer::singleShot(100,this,SLOT(loadData()));



    /*  This code is to test if the neural network actually does learn!
     *  by testing it against XOR problem
     * TODO: move this to it's seperate function

    brain = new NeuralNetwork("ANN", {2,4,4,1});

        H H
      X H H Y
      X H H
        H H



    Matrix * inp = new Matrix(2,1);
    Matrix * tar = new Matrix(1,1);

    int trues = 0;
    int falses = 0;

    for(int i = 0; i < 50000; i++){
        inp->data[0][0] = rand()%2;
        inp->data[1][0] = rand()%2;
        tar->data[0][0] = uchar(inp->data[0][0]) != uchar(inp->data[1][0]) ? 1 : 0;

        int guess = brain->backPropagation(inp, tar)->data[0][0] >= 0.5 ? 1 : 0;

        if(uchar(guess) == uchar(tar->data[0][0])){
            qDebug() << "True";
            trues++;
        }else{
            qDebug() << "False";
            falses++;
        }
    }

    qDebug() << trues << "True   Vs   " << falses << "False";
    */
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
    file.close();
    ba->remove(0,offset);
    if(offset == 16){ // loading an images file
        for(int i = 0; i < ba->size()/784; i++){    // number of images
            QVector<double> *v = new QVector<double>;
            for(int j = 0; j < 784; j++){
                v->append(double(uchar(ba->at(i*784 + j)))/255);
            }
            Matrix *m = new Matrix(v);  delete v;
            vm.append(m);
            load += 784;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54900000 * 100));
        }
    }else{  // loading the labels
        for(int i = 0; i < ba->size(); i++){
            QVector<double> *v = new QVector<double>;
            for(int j = 0; j < 10; j++){
                v->append(uchar(j) == uchar(ba->at(i)) ? 1 : 0);
            }
            Matrix *m = new Matrix(v); delete v;
            vm.append(m);
            load ++;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54900000 * 100));
        }
    }
    delete ba;
}


void MainWindow::viewImage(QVector<Matrix *> &vm, int imgIndex, QLabel *label)
{
    // create uchar array
    uchar data[784];

    for(int i = 0; i < 784; i++){
        data[i] = uchar(vm.at(imgIndex)->data.at(i).at(0) * 255);
    }

    // create QImage & QPixmap
    QImage img(data, 28, 28, QImage::Format_Grayscale8);
    QPixmap pix = QPixmap::fromImage(img);

    // send to view on the Label
    label->setPixmap(pix.scaled(28*5,28*5));
}

void MainWindow::setImageLabel(QVector<Matrix *> &vm, int ind, QLabel *label)
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


void MainWindow::on_testNNBtn_clicked()
{
    if(ui->modelLabel->text() != "Model : undefined"){
        testIndex = 0;
        correctTests = 0;
        QTimer::singleShot(ui->testWaitingSpinBox->value(),this,SLOT(feedImage()));
    }else{
        ui->logPTE->appendPlainText("Warning : No Neural Network Model Configured!");
        ui->logPTE->appendPlainText("          Save or load a model first.");
    }
}

void MainWindow::on_saveModelBtn_clicked()
{
    QString name = ui->modelNameLineEdit->text();
    QVector<int> structure;
    QStringList ls = ui->nnStructLineEdit->text().split(", ",QString::SkipEmptyParts);
    for(int i = 0; i < ls.size(); i++) structure.append(ls[i].toInt());


    brain = new NeuralNetwork(name, structure, ui->batchSizeSpinBox->value(),
                              ui->epochsSpinBox->value(),
                              ui->lrSpinBox->value());
    setNNFullName();
}

QString MainWindow::setNNFullName()
{
    QString str = brain->name;
    str += " NN{";
    for(int i = 0; i < brain->structure.size(); i++){
        str += QString::number(brain->structure[i]);
        if(i < brain->structure.size()-1) str += ", ";
    }
    str += "} ";
    str += "Batch(" + QString::number(brain->batchSize) + ") ";
    str += "Epochs(" + QString::number(brain->epochs) + ") ";
    str += "LR(" + QString::number(brain->learningRate) + ")";
    ui->modelLabel->setText(str);
    return str;
}

void MainWindow::feedImage()
{
    ui->testImgSeekSlider->setValue(testIndex);
    Matrix * output(brain->feedForward(testingImages[testIndex]));

    int guess = 0;
    for(int i = 0; i < output->data.size(); i++){
        if(output->data[guess][0] < output->data[i][0]) guess = i;
    }
    ui->testGuessLabel->setText("Guess : " + QString::number(guess));

    correctTests += int(testingLabels[testIndex]->data[guess][0]);

    double acc = double(correctTests)/testingLabels.size() * 100;
    ui->testAccLabel->setText("Testing Accuracy : " + QString::number(acc,'g',3) + "%  [" +
                              QString::number(correctTests) + "/" +
                              QString::number(testingLabels.size()) + "]");

    ui->testingProgressBar->setValue(int(double(testIndex)/(testingImages.size()-1) * 100));
    if(testIndex < testingImages.size()-1){
        testIndex++;
        QTimer::singleShot(ui->testWaitingSpinBox->value(),this,SLOT(feedImage()));
    }else{
        ui->logPTE->appendPlainText(ui->testAccLabel->text());
    }
}

void MainWindow::on_startTrainingBtn_clicked()
{
    if(ui->modelLabel->text() != "Model : undefined"){
        epochIndex = 0;
        batchIndex = 0;
        trainIndex = 0;
        correctTests = 0;
        brain->shuffleVector(trainingImages, trainingLabels);
        QTimer::singleShot(ui->trainWaitSpinBox->value(),this,SLOT(train()));
    }else{
        ui->logPTE->appendPlainText("Warning : No Neural Network Model Configured!");
        ui->logPTE->appendPlainText("          Save or load a model first.");
    }
}

void MainWindow::train()
{
    ui->trainImagSeekSlider->setValue(trainIndex);
    Matrix * output(brain->backPropagation(trainingImages[trainIndex], trainingLabels[trainIndex]));

    int guess = 0;
    for(int i = 0; i < output->data.size(); i++){
        if(output->data[guess][0] < output->data[i][0]) guess = i;
    }
    ui->trainGuessLabel->setText("Guess : " + QString::number(guess));

    correctTests += int(trainingLabels[trainIndex]->data[guess][0]);

    double acc = double(correctTests)/trainingLabels.size() * 100;
    ui->trainAccLabel->setText("Training : Epoch["+QString::number(epochIndex)+"]  Batch["+
                QString::number(trainIndex/brain->batchSize) + "/" +
                QString::number(trainingLabels.size()/brain->batchSize)+
                "]  Accuracy : " + QString::number(acc,'g',3) + "%  [" +
                QString::number(correctTests) + "/" +
                QString::number(trainingLabels.size()) + "]");

    ui->traingingProgressBar->setValue(int(double(trainIndex)/(trainingImages.size()-1) * 100));


    trainIndex++;
    batchIndex++;

    if(batchIndex >= brain->batchSize-1){
        batchIndex = 0;
        for(int i = 0; i < brain->weights.size(); i++){
            // divide the already summed gradients
            //   and deltas by batchSize to get Average
            //brain->gradients[i]->divide(brain->batchSize);
            //brain->deltas[i]->divide(brain->batchSize);

            // Adjust weights and biases
            //brain->biases[i]->add(brain->gradients[i]);
            //brain->weights[i]->add(brain->deltas[i]);

            // reset gradients and deltas?
            //brain->gradients[i]->multiply(0);
            //brain->deltas[i]->multiply(0);
        }
    }

    if(trainIndex >= trainingImages.size()-1){
        trainIndex = 0;
        correctTests = 0;
        batchIndex = 0;
        epochIndex++;
        brain->shuffleVector(trainingImages, trainingLabels);
        ui->logPTE->appendPlainText(ui->trainAccLabel->text());
    }

    if(epochIndex < brain->epochs){
        QTimer::singleShot(ui->trainWaitSpinBox->value(),this,SLOT(train()));
    }
}

QString MainWindow::getRandomName(int len)
{
    QString str;
    for(int i = 0; i < len; i++){
        str += char(rand()%26 + 65);
    }
    return  str;
}
