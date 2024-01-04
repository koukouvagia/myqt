#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QFileDialog>
#include <QImage>
#include <QLabel>
#include <QMainWindow>
#include <QPixmap>
#include <QString>
#include <QWidget>

#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <deque>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Ui { class MainWindow; }

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // API
    cv::Mat scale(const cv::Mat &src, int w, int h);
    cv::Mat scale(const cv::Mat &src, float r);
    cv::Mat rotate(const cv::Mat &src, int type);
    cv::Mat mirror(const cv::Mat &src, int type);

    cv::Mat brightness(const cv::Mat &src, int n);
    cv::Mat contrast(const cv::Mat &src, int n);
    cv::Mat exposure(const cv::Mat &src, int n);
    cv::Mat highlight(const cv::Mat &src, int n);
    cv::Mat shadow(const cv::Mat &src, int n);
    cv::Mat saturation(const cv::Mat &src, int n);
    cv::Mat warmth(const cv::Mat &src, int n);
    cv::Mat tint(const cv::Mat &src, int n);
    cv::Mat sharpness(const cv::Mat &src, int n);
    cv::Mat smoothness(const cv::Mat &src, int n);

    cv::Mat noiseGaussian(const cv::Mat &src, int n);
    cv::Mat noisePoisson(const cv::Mat &src, int n);
    cv::Mat noiseImpulse(const cv::Mat &src, int n);
    cv::Mat noiseSpeckle(const cv::Mat &src, int n);
    cv::Mat blurDefocus(const cv::Mat &src, int n);
    cv::Mat blurMotion(const cv::Mat &src, int n);
    cv::Mat blurZoom(const cv::Mat &src, int n);

    // Aux
    QImage matToImg(const cv::Mat &mat);
    cv::Mat imgToMat(const QImage &img);
    int channelNum(const cv::Mat &mat);
    int channelNum(const QImage &img);

    void showPicture(const QString &path);
    void showPicture(const cv::Mat &mat);
    void showChannel(const cv::Mat &mat);
    void showMatrix(const char *name, const cv::Mat &mat);

    void cancelChanges();
    void applyChanges();
    void resetValuesE();
    void resetValuesD();

private:
    Ui::MainWindow *ui;

    cv::Mat srcMat, dstMat;
    QString openPath, savePath;
    std::deque<cv::Mat> undoLog, redoLog;

public slots:
    void onButtonOpenClicked();
    void onButtonSaveClicked();
    void onButtonUndoClicked();
    void onButtonRedoClicked();

    void onButtonCropClicked(); // TODO
    void onButtonCropSClicked(); // TODO
    void onButtonCropRClicked(); // TODO
    void onButtonRotateAClicked();
    void onButtonRotateCClicked();
    void onButtonMirrorHClicked();
    void onButtonMirrorVClicked();
    void onButtonCancelGClicked();
    void onButtonApplyGClicked();

    void onSliderEnhanceValuesChanged();
    void onSpinEnhanceValuesChanged();
    void onButtonCancelEClicked();
    void onButtonApplyEClicked();

    void onSliderDegradeValuesChanged(); // TODO
    void onSpinDegradeValuesChanged(); // TODO
    void onButtonCancelDClicked();
    void onButtonApplyDClicked();

signals:

};

#endif // MAINWINDOW_H
