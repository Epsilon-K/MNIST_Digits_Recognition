#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <NeuralNetwork.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    NeuralNetwork *brain;
};

#endif // MAINWINDOW_H
