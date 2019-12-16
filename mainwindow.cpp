#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    loadStyle();
    srand(time(0));

    ui->modelNameLineEdit->setText(getRandomName(3));
    realTime = ui->checkBox->isChecked();
    QTimer::singleShot(200,this,SLOT(loadData()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::loadDataSet(QString path, QVector<QVector<uchar>> &data, int offset)
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
            QVector<uchar> img;
            for(int j = 0; j < 784; j++){
                img.append(uchar(ba->at(i*784 + j)));
            }
            data.append(img);
            load += 784;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54950000 * 100));
        }
    }else{  // loading the labels
        for(int i = 0; i < ba->size(); i++){
            QVector<uchar> lbl;
            for(int j = 0; j < 10; j++){
                lbl.append(uchar(ba->at(i)));
            }
            data.append(lbl);
            load ++;
            if(int(load) % 1000 == 0)
                ui->loadingProgressBar->setValue(int(load/54950000 * 100));
        }
    }
    delete ba;
}


void MainWindow::viewImage(QVector<QVector<uchar>> &data, int imgIndex, QLabel *label)
{
    // create QImage & QPixmap
    QImage img(data[imgIndex].data(), 28, 28, QImage::Format_Grayscale8);
    QPixmap pix = QPixmap::fromImage(img);

    // send to view on the Label
    label->setPixmap(pix.scaled(28*5,28*5));
}

void MainWindow::setImageLabel(QVector<QVector<uchar>> &data, int ind, QLabel *label)
{
    QString str = "image[" + QString::number(ind) + "] Label : ";
    str += QString::number(data[ind][0]);
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
    deleteDataLoadingBar();
    QTimer::singleShot(20,this,SLOT(showMaximized()));
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
    // save to disk!!
    QString fileName = QFileDialog::getSaveFileName
            (this, "Save ANN Model", QDir::currentPath(),
            "Artificial Neural Network file (*.ann)");
    brain->path = fileName;
    save();
}

QString MainWindow::setNNFullName()
{
    QString str = brain->name;
    str += " {";
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
    Matrix *input = new Matrix(testingImages[testIndex]);
    input->divide(255); // normalize;
    Matrix * output(brain->feedForward(input)); delete input;

    int guess = 0;
    for(int i = 0; i < output->data.size(); i++){
        if(output->data[guess][0] < output->data[i][0]) guess = i;
    }
    ui->testGuessLabel->setText("Guess : " + QString::number(guess));

    correctTests += guess == testingLabels.at(testIndex).at(0) ? 1 : 0;

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
        tiem.start();
        ui->logPTE->appendPlainText("Starting Epoch[1/" + QString::number(brain->epochs)
                                    + "] at : " + tiem.currentTime().toString());

        QTimer::singleShot(ui->trainWaitSpinBox->value(),this,SLOT(train()));
    }else{
        ui->logPTE->appendPlainText("Warning : No Neural Network Model Configured!");
        ui->logPTE->appendPlainText("          Save or load a model first.");
    }
}

void MainWindow::train()
{
    if(realTime) ui->trainImagSeekSlider->setValue(trainIndex);
    Matrix *input = new Matrix(trainingImages[trainIndex]);
    input->divide(255); // normalize;
    Matrix * targets = new Matrix(10,1);
    for(int i = 0; i < 10; i++){targets->data[i][0] = i == int(trainingLabels[trainIndex][0]) ? 1 : 0;}
    Matrix * output(brain->backPropagation(input, targets)); delete input; delete targets;

    int guess = 0;
    for(int i = 0; i < output->data.size(); i++){
        if(output->data[guess][0] < output->data[i][0]) guess = i;
    }
    ui->trainGuessLabel->setText("Guess : " + QString::number(guess));

    correctTests += guess == trainingLabels.at(trainIndex).at(0) ? 1 : 0;

    double acc = double(correctTests)/trainingLabels.size() * 100;
    ui->trainAccLabel->setText("Training : Epoch["+QString::number(epochIndex+1)+"/"+
                QString::number(brain->epochs) + "]  Batch["+
                QString::number((trainIndex/brain->batchSize) + 1) + "/" +
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
            brain->gradients[i]->divide(brain->batchSize);
            brain->deltas[i]->divide(brain->batchSize);

            // Adjust weights and biases
            brain->biases[i]->add(brain->gradients[i]);
            brain->weights[i]->add(brain->deltas[i]);

            // reset gradients and deltas?
            brain->gradients[i]->multiply(0);
            brain->deltas[i]->multiply(0);
        }
    }

    if(trainIndex >= trainingImages.size()-1){
        trainIndex = 0;
        correctTests = 0;
        batchIndex = 0;
        epochIndex++;
        brain->shuffleVector(trainingImages, trainingLabels);
        ui->logPTE->appendPlainText(ui->trainAccLabel->text());
        ui->logPTE->appendPlainText("Epoch[" +QString::number(epochIndex) + "/" +
        QString::number(brain->epochs) + "] Ended at : " + tiem.currentTime().toString() +
        "   -   elapsed time : " + QString::number(double(tiem.restart())/1000/60, 'g', 3) + " minutes");
        save();
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

void MainWindow::save()
{
    QFile file(brain->path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)){
        ui->logPTE->appendPlainText("Cannot write to file : " + brain->path);
        return;
    }

    QTextStream out(&file);
    out << brain->toString();
    file.close();
}

void MainWindow::proof()
{
    /*  This code is to test if the neural network actually does learn!
     *  by testing it against XOR problem
     */

    NeuralNetwork *br = new NeuralNetwork(QDir::currentPath(), "ANN", {2,4,4,1});

    Matrix * inp = new Matrix(2,1);
    Matrix * tar = new Matrix(1,1);

    int trues = 0;
    int falses = 0;

    for(int i = 0; i < 50000; i++){
        inp->data[0][0] = rand()%2;
        inp->data[1][0] = rand()%2;
        tar->data[0][0] = uchar(inp->data[0][0]) != uchar(inp->data[1][0]) ? 1 : 0;

        int guess = br->backPropagation(inp, tar)->data[0][0] >= 0.5 ? 1 : 0;

        if(uchar(guess) == uchar(tar->data[0][0])){
            qDebug() << "True";
            trues++;
        }else{
            qDebug() << "False";
            falses++;
        }
    }

    qDebug() << trues << "True   Vs   " << falses << "False";
    delete br;
    delete inp;
    delete tar;
}

void MainWindow::on_createModelBtn_clicked()
{
    QString name = ui->modelNameLineEdit->text();
    QVector<int> structure;
    QStringList ls = ui->nnStructLineEdit->text().split(", ",QString::SkipEmptyParts);
    for(int i = 0; i < ls.size(); i++) structure.append(ls[i].toInt());

    QString fileName = QFileDialog::getSaveFileName
            (this, "Save ANN Model", QDir::currentPath(),
            "Artificial Neural Network file (*.ann)");

    NeuralNetwork * newBrain = new NeuralNetwork(fileName, name, structure, ui->batchSizeSpinBox->value(),
                              ui->epochsSpinBox->value(),
                              ui->lrSpinBox->value());

    if(ui->modelLabel->text() != "Model : undefined"){   // is there an old NN?
        if(brain->isSameStructure(newBrain)){  // is it the same structure?
            // copy weights and biases?
            QMessageBox * mb = new QMessageBox(this);
            mb->setText("New Neural Network");
            mb->setInformativeText("Copy Weights and Biases to new Network?");
            mb->setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            mb->setDefaultButton(QMessageBox::Yes);
            switch (mb->exec()) {
            case QMessageBox::Yes :
                newBrain->copyWnB(brain);
            break;
            case QMessageBox::No :
                // move on
            break;
            case QMessageBox::Cancel :
            default:
                return;
            }
        }
        delete brain;
    }

    brain = newBrain;
    save();
    setNNFullName();
}

void MainWindow::on_loadModelSpinBox_clicked()
{
    QString fileName = QFileDialog::getOpenFileName(this,
          "Open ANN file", QDir::currentPath(), "Artificial Neural Network file (*.ann)");
    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
        ui->logPTE->appendPlainText("can't open file : " + fileName);
        return;
    }

    NeuralNetwork * newBrain = NeuralNetwork::fromString(file.readAll());
    if(ui->modelLabel->text() != "Model : undefined"){   // is there an old NN?
        if(brain->isSameStructure(newBrain)){  // is it the same structure?
            // copy weights and biases?
            QMessageBox * mb = new QMessageBox(this);
            mb->setText("New Neural Network");
            mb->setInformativeText("Copy Weights and Biases to new Network?");
            mb->setStandardButtons(QMessageBox::Yes | QMessageBox::No);
            mb->setDefaultButton(QMessageBox::Yes);
            switch (mb->exec()) {
            case QMessageBox::Yes :
                newBrain->copyWnB(brain);
            break;
            case QMessageBox::No :
                // move on
            break;
            case QMessageBox::Cancel :
            default:
                return;
            }
        }
        delete brain;
    }

    brain = newBrain;
    setNNFullName();
}

void MainWindow::deleteDataLoadingBar()
{
    ui->loadingLabel->deleteLater();
    ui->loadingProgressBar->deleteLater();
}

void MainWindow::on_checkBox_toggled(bool checked)
{
    realTime = checked;
}
