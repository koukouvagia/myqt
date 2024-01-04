#include "mainwindow.h"
#include "ui_mainwindow.h"

#define LOGSIZE 20

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    openPath = savePath = QDir::homePath();

    // top bar
    connect(ui->buttonOpen, SIGNAL(clicked()), this, SLOT(onButtonOpenClicked()));
    connect(ui->buttonSave, SIGNAL(clicked()), this, SLOT(onButtonSaveClicked()));
    connect(ui->buttonUndo, SIGNAL(clicked()), this, SLOT(onButtonUndoClicked()));
    connect(ui->buttonRedo, SIGNAL(clicked()), this, SLOT(onButtonRedoClicked()));

    // page: graphic
    connect(ui->buttonRotateA, SIGNAL(clicked()), this, SLOT(onButtonRotateAClicked()));
    connect(ui->buttonRotateC, SIGNAL(clicked()), this, SLOT(onButtonRotateCClicked()));
    connect(ui->buttonMirrorH, SIGNAL(clicked()), this, SLOT(onButtonMirrorHClicked()));
    connect(ui->buttonMirrorV, SIGNAL(clicked()), this, SLOT(onButtonMirrorVClicked()));

    connect(ui->buttonCancelG, SIGNAL(clicked()), this, SLOT(onButtonCancelGClicked()));
    connect(ui->buttonApplyG, SIGNAL(clicked()), this, SLOT(onButtonApplyGClicked()));

    // page: enhance
    connect(ui->sliderEnhanceBr, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceBr, SIGNAL(valueChanged(int)), ui->spinEnhanceBr, SLOT(setValue(int)));
    connect(ui->spinEnhanceBr, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceBr, SIGNAL(valueChanged(int)), ui->sliderEnhanceBr, SLOT(setValue(int)));

    connect(ui->sliderEnhanceCt, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceCt, SIGNAL(valueChanged(int)), ui->spinEnhanceCt, SLOT(setValue(int)));
    connect(ui->spinEnhanceCt, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceCt, SIGNAL(valueChanged(int)), ui->sliderEnhanceCt, SLOT(setValue(int)));

    connect(ui->sliderEnhanceEx, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceEx, SIGNAL(valueChanged(int)), ui->spinEnhanceEx, SLOT(setValue(int)));
    connect(ui->spinEnhanceEx, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceEx, SIGNAL(valueChanged(int)), ui->sliderEnhanceEx, SLOT(setValue(int)));

    connect(ui->sliderEnhanceHl, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceHl, SIGNAL(valueChanged(int)), ui->spinEnhanceHl, SLOT(setValue(int)));
    connect(ui->spinEnhanceHl, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceHl, SIGNAL(valueChanged(int)), ui->sliderEnhanceHl, SLOT(setValue(int)));

    connect(ui->sliderEnhanceSd, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceSd, SIGNAL(valueChanged(int)), ui->spinEnhanceSd, SLOT(setValue(int)));
    connect(ui->spinEnhanceSd, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceSd, SIGNAL(valueChanged(int)), ui->sliderEnhanceSd, SLOT(setValue(int)));

    connect(ui->sliderEnhanceSt, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceSt, SIGNAL(valueChanged(int)), ui->spinEnhanceSt, SLOT(setValue(int)));
    connect(ui->spinEnhanceSt, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceSt, SIGNAL(valueChanged(int)), ui->sliderEnhanceSt, SLOT(setValue(int)));

    connect(ui->sliderEnhanceWm, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceWm, SIGNAL(valueChanged(int)), ui->spinEnhanceWm, SLOT(setValue(int)));
    connect(ui->spinEnhanceWm, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceWm, SIGNAL(valueChanged(int)), ui->sliderEnhanceWm, SLOT(setValue(int)));

    connect(ui->sliderEnhanceTn, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceTn, SIGNAL(valueChanged(int)), ui->spinEnhanceTn, SLOT(setValue(int)));
    connect(ui->spinEnhanceTn, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceTn, SIGNAL(valueChanged(int)), ui->sliderEnhanceTn, SLOT(setValue(int)));

    connect(ui->sliderEnhanceSp, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceSp, SIGNAL(valueChanged(int)), ui->spinEnhanceSp, SLOT(setValue(int)));
    connect(ui->spinEnhanceSp, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceSp, SIGNAL(valueChanged(int)), ui->sliderEnhanceSp, SLOT(setValue(int)));

    connect(ui->sliderEnhanceSm, SIGNAL(valueChanged(int)), this, SLOT(onSliderEnhanceValuesChanged()));
    connect(ui->sliderEnhanceSm, SIGNAL(valueChanged(int)), ui->spinEnhanceSm, SLOT(setValue(int)));
    connect(ui->spinEnhanceSm, SIGNAL(valueChanged(int)), this, SLOT(onSpinEnhanceValuesChanged()));
    connect(ui->spinEnhanceSm, SIGNAL(valueChanged(int)), ui->sliderEnhanceSm, SLOT(setValue(int)));

    connect(ui->buttonCancelE, SIGNAL(clicked()), this, SLOT(onButtonCancelEClicked()));
    connect(ui->buttonApplyE, SIGNAL(clicked()), this, SLOT(onButtonApplyEClicked()));

    // page: degrade
    connect(ui->sliderDegradeNg, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeNg, SIGNAL(valueChanged(int)), ui->spinDegradeNg, SLOT(setValue(int)));
    connect(ui->spinDegradeNg, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeNg, SIGNAL(valueChanged(int)), ui->sliderDegradeNg, SLOT(setValue(int)));

    connect(ui->sliderDegradeNp, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeNp, SIGNAL(valueChanged(int)), ui->spinDegradeNp, SLOT(setValue(int)));
    connect(ui->spinDegradeNp, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeNp, SIGNAL(valueChanged(int)), ui->sliderDegradeNp, SLOT(setValue(int)));

    connect(ui->sliderDegradeNi, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeNi, SIGNAL(valueChanged(int)), ui->spinDegradeNi, SLOT(setValue(int)));
    connect(ui->spinDegradeNi, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeNi, SIGNAL(valueChanged(int)), ui->sliderDegradeNi, SLOT(setValue(int)));

    connect(ui->sliderDegradeNs, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeNs, SIGNAL(valueChanged(int)), ui->spinDegradeNs, SLOT(setValue(int)));
    connect(ui->spinDegradeNs, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeNs, SIGNAL(valueChanged(int)), ui->sliderDegradeNs, SLOT(setValue(int)));

    connect(ui->sliderDegradeBd, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeBd, SIGNAL(valueChanged(int)), ui->spinDegradeBd, SLOT(setValue(int)));
    connect(ui->spinDegradeBd, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeBd, SIGNAL(valueChanged(int)), ui->sliderDegradeBd, SLOT(setValue(int)));

    connect(ui->sliderDegradeBm, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeBm, SIGNAL(valueChanged(int)), ui->spinDegradeBm, SLOT(setValue(int)));
    connect(ui->spinDegradeBm, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeBm, SIGNAL(valueChanged(int)), ui->sliderDegradeBm, SLOT(setValue(int)));

    connect(ui->sliderDegradeBz, SIGNAL(valueChanged(int)), this, SLOT(onSliderDegradeValuesChanged()));
    connect(ui->sliderDegradeBz, SIGNAL(valueChanged(int)), ui->spinDegradeBz, SLOT(setValue(int)));
    connect(ui->spinDegradeBz, SIGNAL(valueChanged(int)), this, SLOT(onSpinDegradeValuesChanged()));
    connect(ui->spinDegradeBz, SIGNAL(valueChanged(int)), ui->sliderDegradeBz, SLOT(setValue(int)));

    connect(ui->buttonCancelD, SIGNAL(clicked()), this, SLOT(onButtonCancelDClicked()));
    connect(ui->buttonApplyD, SIGNAL(clicked()), this, SLOT(onButtonApplyDClicked()));

    // page: restore

}

MainWindow::~MainWindow()
{
    delete ui;
}

/**************** API ****************/

/* page: graphic */

cv::Mat MainWindow::scale(const cv::Mat &src, int w, int h)
{
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(w, h));
    return dst;
}

cv::Mat MainWindow::scale(const cv::Mat &src, float r)
{
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(src.cols * r, src.rows * r));
    return dst;
}

cv::Mat MainWindow::rotate(const cv::Mat &src, int type)
{
    cv::Mat dst;
    if (type == 1) {
        cv::rotate(src, dst, 2); // anticlock
    } else if (type == 0) {
        cv::rotate(src, dst, 0); // clockwise
    }
    return dst;
}

cv::Mat MainWindow::mirror(const cv::Mat &src, int type)
{
    cv::Mat dst;
    if (type == 1) {
        cv::flip(src, dst, 1); // horizontal
    } else if (type == 0) {
        cv::flip(src, dst, 0); // vertical
    }
    return dst;
}

/* page: enhance */

cv::Mat MainWindow::brightness(const cv::Mat &src, int n)
{
    float alpha = (float)n / 400.f;
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            if (alpha >= 0) {
                d[ch * j] = cv::saturate_cast<uchar>(s[ch * j] * (1.f - alpha) + 255 * alpha);
                d[ch * j + 1] = cv::saturate_cast<uchar>(s[ch * j + 1] * (1.f - alpha) + 255 * alpha);
                d[ch * j + 2] = cv::saturate_cast<uchar>(s[ch * j + 2] * (1.f - alpha) + 255 * alpha);
            } else {
                d[ch * j] = cv::saturate_cast<uchar>(s[ch * j] * (1.f + alpha));
                d[ch * j + 1] = cv::saturate_cast<uchar>(s[ch * j + 1] * (1.f + alpha));
                d[ch * j + 2] = cv::saturate_cast<uchar>(s[ch * j + 2] * (1.f + alpha));
            }
        }
    }
    return dst;
}

cv::Mat MainWindow::contrast(const cv::Mat &src, int n)
{
    float percent = (float)n / 200.f;
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            if (percent >= 0) {
                d[ch * j] = cv::saturate_cast<uchar>((s[ch * j] - 127) / (1.f - percent) + 127);
                d[ch * j + 1] = cv::saturate_cast<uchar>((s[ch * j + 1] - 127) / (1.f - percent) + 127);
                d[ch * j + 2] = cv::saturate_cast<uchar>((s[ch * j + 2] - 127) / (1.f - percent) + 127);
            } else {
                d[ch * j] = cv::saturate_cast<uchar>((s[ch * j] - 127) * (1.f + percent) + 127);
                d[ch * j + 1] = cv::saturate_cast<uchar>((s[ch * j + 1] - 127) / (1.f - percent) + 127);
                d[ch * j + 2] = cv::saturate_cast<uchar>((s[ch * j + 2] - 127) / (1.f - percent) + 127);
            }
        }
    }
    return dst;
}

cv::Mat MainWindow::exposure(const cv::Mat &src, int n)
{
    float gamma = pow(10, -(float)n / 400.f);
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            d[ch * j] = cv::saturate_cast<uchar>(pow(s[ch * j] / 255.f, gamma) * 255);
            d[ch * j + 1] = cv::saturate_cast<uchar>(pow(s[ch * j + 1] / 255.f, gamma) * 255);
            d[ch * j + 2] = cv::saturate_cast<uchar>(pow(s[ch * j + 2] / 255.f, gamma) * 255);
        }
    }
    return dst;
}

cv::Mat MainWindow::highlight(const cv::Mat &src, int n)
{
    float bright = (float)n / 800.f;
    float dark = 1.f + (float)n / 800.f;
    cv::Mat srcCopy = src.clone();
    srcCopy.convertTo(srcCopy, CV_32FC3);
    std::vector<cv::Mat> bgra;
    cv::split(srcCopy, bgra);
    cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
    gray = 0.299f * bgra[2] + 0.587f * bgra[1] + 0.114f * bgra[0];
    gray = gray / 255.f;
    cv::Mat thresh = cv::Mat::zeros(src.size(), CV_32FC1);
    thresh = gray.mul(gray);
    cv::Scalar t = mean(thresh);
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    mask.setTo(255, thresh >= t[0]);
    cv::Mat bRate = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat dRate = cv::Mat::zeros(src.size(), CV_32FC1);
    for (int i = 0; i < src.rows; ++i)
    {
        uchar* m = mask.ptr<uchar>(i);
        float* th = thresh.ptr<float>(i);
        float* br = bRate.ptr<float>(i);
        float* dr = dRate.ptr<float>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            br[j] = m[j] == 255 ? bright : bright * th[j] / t[0];
            dr[j] = m[j] == 255 ? dark : 1.f + (dark - 1.f) * th[j] / t[0];
        }
    }
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        float* br = bRate.ptr<float>(i);
        float* dr = dRate.ptr<float>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            d[ch * j] = cv::saturate_cast<uchar>(pow(s[ch * j] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
            d[ch * j + 1] = cv::saturate_cast<uchar>(pow(s[ch * j + 1] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
            d[ch * j + 2] = cv::saturate_cast<uchar>(pow(s[ch * j + 2] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
        }
    }
    return dst;
}

cv::Mat MainWindow::shadow(const cv::Mat &src, int n)
{
    float bright = (float)n / 800.f;
    float dark = 1.f + (float)n / 800.f;
    cv::Mat srcCopy = src.clone();
    srcCopy.convertTo(srcCopy, CV_32FC3);
    std::vector<cv::Mat> bgra;
    cv::split(srcCopy, bgra);
    cv::Mat gray = cv::Mat::zeros(src.size(), CV_32FC1);
    gray = 0.299f * bgra[2] + 0.587f * bgra[1] + 0.114f * bgra[0];
    gray = gray / 255.f;
    cv::Mat thresh = cv::Mat::zeros(src.size(), CV_32FC1);
    thresh = (1.f - gray).mul(1.f - gray);
    cv::Scalar t = mean(thresh);
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_8UC1);
    mask.setTo(255, thresh >= t[0]);
    cv::Mat bRate = cv::Mat::zeros(src.size(), CV_32FC1);
    cv::Mat dRate = cv::Mat::zeros(src.size(), CV_32FC1);
    for (int i = 0; i < src.rows; ++i)
    {
        uchar* m = mask.ptr<uchar>(i);
        float* th = thresh.ptr<float>(i);
        float* br = bRate.ptr<float>(i);
        float* dr = dRate.ptr<float>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            br[j] = m[j] == 255 ? bright : bright * th[j] / t[0];
            dr[j] = m[j] == 255 ? dark : 1.f + (dark - 1.f) * th[j] / t[0];
        }
    };
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        float* br = bRate.ptr<float>(i);
        float* dr = dRate.ptr<float>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            d[ch * j] = cv::saturate_cast<uchar>(pow(s[ch * j] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
            d[ch * j + 1] = cv::saturate_cast<uchar>(pow(s[ch * j + 1] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
            d[ch * j + 2] = cv::saturate_cast<uchar>(pow(s[ch * j + 2] / 255.f, 1.f / dr[j]) * (1.f / (1.f - br[j])) * 255);
        }
    }
    return dst;
}

cv::Mat MainWindow::saturation(const cv::Mat &src, int n)
{
    float percent = (float)n / 200.f;
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            float max = std::max(s[ch * j], std::max(s[ch * j + 1], s[ch * j + 2]));
            float min = std::min(s[ch * j], std::min(s[ch * j + 1], s[ch * j + 2]));
            float delta = (max - min) / 255.f;
            if (delta == 0) continue;
            float value = (max + min) / 255.f;
            float L = value / 2.f;
            float S = L < 0.5 ? delta / value : delta / (2.f - value);
            float alpha;
            if (percent >= 0) {
                alpha = percent + S >= 1 ? S : 1.f - percent;
                alpha = 1.f / alpha - 1.f;
                d[ch * j] = cv::saturate_cast<uchar>(s[ch * j] + (s[ch * j] - L * 255) * alpha);
                d[ch * j + 1] = cv::saturate_cast<uchar>(s[ch * j + 1] + (s[ch * j + 1] - L * 255) * alpha);
                d[ch * j + 2] = cv::saturate_cast<uchar>(s[ch * j + 2] + (s[ch * j + 2] - L * 255) * alpha);
            } else {
                alpha = percent;
                d[ch * j] = cv::saturate_cast<uchar>(L * 255 + (s[ch * j] - L * 255) * (1.f + alpha));
                d[ch * j + 1] = cv::saturate_cast<uchar>(L * 255 + (s[ch * j + 1] - L * 255) * (1.f + alpha));
                d[ch * j + 2] = cv::saturate_cast<uchar>(L * 255 + (s[ch * j + 2] - L * 255) * (1.f + alpha));
            }
        }
    }
    return dst;
}

cv::Mat MainWindow::warmth(const cv::Mat &src, int n)
{
    int level = n / 4;
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            d[ch * j] = cv::saturate_cast<uchar>(s[ch * j] - level);
            d[ch * j + 1] = cv::saturate_cast<uchar>(s[ch * j + 1] + level);
            d[ch * j + 2] = cv::saturate_cast<uchar>(s[ch * j + 2] + level);
        }
    }
    return dst;
}

cv::Mat MainWindow::tint(const cv::Mat &src, int n)
{
    int level = n / 4;
    cv::Mat dst = src.clone();
    int ch = channelNum(src);
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            d[ch * j] = cv::saturate_cast<uchar>(s[ch * j]);
            d[ch * j + 1] = cv::saturate_cast<uchar>(s[ch * j + 1] - level);
            d[ch * j + 2] = cv::saturate_cast<uchar>(s[ch * j + 2] + level);
        }
    }
    return dst;
}

cv::Mat MainWindow::sharpness(const cv::Mat &src, int n)
{
    float percent = (float)n / 100.f;
    cv::Mat dst;
    cv::Laplacian(src, dst, src.depth(), 1);
    dst = src - dst * percent;
    return dst;
}

cv::Mat MainWindow::smoothness(const cv::Mat &src, int n)
{
    float percent = (float)n / 100.f;
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(7, 7), 0);
    dst = src * (1.f - percent) + dst * percent;
    return dst;
}

/* page: degrade */

cv::Mat MainWindow::noiseGaussian(const cv::Mat &src, int n)
{
    float sigma = (float)n / 2.f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.f, sigma);
    cv::Mat dst = src.clone();
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            float prob = dis(gen);
            d[3 * j] = cv::saturate_cast<uchar>(s[3 * j] + prob);
            d[3 * j + 1] = cv::saturate_cast<uchar>(s[3 * j + 1] + prob);
            d[3 * j + 2] = cv::saturate_cast<uchar>(s[3 * j + 2] + prob);
        }
    }
    return dst;
}

cv::Mat MainWindow::noisePoisson(const cv::Mat &src, int n)
{
    int lambda = n / 2 + 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<int> dis(lambda);
    cv::Mat dst = src.clone();
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            int prob = dis(gen);
            d[3 * j] = cv::saturate_cast<uchar>(s[3 * j] + prob);
            d[3 * j + 1] = cv::saturate_cast<uchar>(s[3 * j + 1] + prob);
            d[3 * j + 2] = cv::saturate_cast<uchar>(s[3 * j + 2] + prob);
        }
    }
    return dst;
}

cv::Mat MainWindow::noiseImpulse(const cv::Mat &src, int n)
{
    float level = (float)n / 200.f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.f, 1.f);
    cv::Mat dst = src.clone();
    for (int i = 0; i < src.rows; ++i)
    {
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            float prob = dis(gen);
            if (prob < level / 4.f)
            {
                d[3 * j] = d[3 * j + 1] = d[3 * j + 2] = 0;
            }
            else if (prob < level / 2.f)
            {
                d[3 * j] = d[3 * j + 1] = d[3 * j + 2] = 255;
            }
        }
    }
    return dst;
}

cv::Mat MainWindow::noiseSpeckle(const cv::Mat &src, int n)
{
    float sigma = (float)n / 200.f;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.f, sigma);
    cv::Mat dst = src.clone();
    for (int i = 0; i < src.rows; ++i)
    {
        const uchar* s = src.ptr<const uchar>(i);
        uchar* d = dst.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            float prob = dis(gen);
            d[3 * j] = cv::saturate_cast<uchar>(s[3 * j] + s[3 * j] * prob);
            d[3 * j + 1] = cv::saturate_cast<uchar>(s[3 * j + 1] + s[3 * j + 1] * prob);
            d[3 * j + 2] = cv::saturate_cast<uchar>(s[3 * j + 2] + s[3 * j + 2] * prob);
        }
    }
    return dst;
}

cv::Mat MainWindow::blurDefocus(const cv::Mat &src, int n)
{
    int radius = n / 4;
    cv::Mat srcCopy = src.clone();
    srcCopy.convertTo(srcCopy, CV_32FC3, 1.f / 255.f);
    std::vector<int> range;
    for (int i = -radius; i <= radius; ++i) range.push_back(i);
    cv::Mat x, y;
    cv::repeat(cv::Mat(range).t(), 2 * radius + 1, 1, x);
    cv::repeat(cv::Mat(range), 1, 2 * radius + 1, y);
    cv::Mat kernel = cv::Mat::zeros(cv::Size(2 * radius + 1, 2 * radius + 1), CV_32FC1);
    kernel.setTo(1.f, x.mul(x) + y.mul(y) <= radius * radius);
    kernel /= cv::sum(kernel);
    cv::GaussianBlur(kernel, kernel, cv::Size(3, 3), 0.1);
    cv::Mat dst;
    cv::filter2D(srcCopy, dst, -1, kernel);
    dst.convertTo(dst, src.type(), 255);
    return dst;
}

cv::Mat MainWindow::blurMotion(const cv::Mat &src, int n)
{
    int ksize = n / 2 + 1;
    cv::Mat srcCopy = src.clone();
    srcCopy.convertTo(srcCopy, CV_32FC3, 1.f / 255.f);
    cv::Mat mat = cv::getRotationMatrix2D(cv::Point2f(ksize / 2, ksize / 2), 45, 1);
    cv::Mat kernel = cv::Mat::eye(ksize, ksize, CV_32FC1);
    cv::warpAffine(kernel, kernel, mat, cv::Size(ksize, ksize));
    kernel /= (float)ksize;
    cv::Mat dst;
    cv::filter2D(srcCopy, dst, -1, kernel);
    dst.convertTo(dst, src.type(), 255);
    return dst;
}

cv::Mat MainWindow::blurZoom(const cv::Mat &src, int n)
{
    int iteration = n / 4;
    cv::Mat srcCopy = src.clone();
    srcCopy.convertTo(srcCopy, CV_32FC3, 1.f / 255.f);
    cv::Mat dst = srcCopy.clone();
    float end = 1.f + (float)iteration / 100.f;
    for (float ratio = 1.f; ratio < end; ratio += 0.01f)
    {
        int w = (float)src.cols / ratio, h = (float)src.rows / ratio;
        int x = (src.cols - w) / 2, y = (src.rows - h) / 2;
        cv::Mat roi = srcCopy(cv::Range(y, y + h), cv::Range(x, x + w));
        cv::resize(roi, roi, src.size());
        dst += roi;
    }
    dst /= ((float)iteration + 1.f);
    dst.convertTo(dst, src.type(), 255);
    return dst;
}

/**************** Aux ****************/

QImage MainWindow::matToImg(const cv::Mat &mat)
{
    QImage img;
    switch(mat.type())
    {
    case CV_8UC1:
        img = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
        break;
    case CV_8UC3: // 16
        img = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        img = img.rgbSwapped();
        break;
    case CV_8UC4: // 24
        img = QImage((const uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
        break;
    default:
        break;
    }
    return img;
}

cv::Mat MainWindow::imgToMat(const QImage &img)
{
    cv::Mat mat;
    switch(img.format())
    {
    case QImage::Format_Indexed8: // 3
    case QImage::Format_Grayscale8: // 24
        mat = cv::Mat(img.height(), img.width(), CV_8UC1, (void*)img.constBits(), img.bytesPerLine());
        break;
    case QImage::Format_RGB888: // 13
        mat = cv::Mat(img.height(), img.width(), CV_8UC3, (void*)img.constBits(), img.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        break;
    case QImage::Format_RGB32: // 4, 0xffRRGGBB
    case QImage::Format_ARGB32: // 5, 0xAARRGGBB
    case QImage::Format_ARGB32_Premultiplied: // 6
        mat = cv::Mat(img.height(), img.width(), CV_8UC4, (void*)img.constBits(), img.bytesPerLine());
        break;
    default:
        break;
    }
    return mat;
}

int MainWindow::channelNum(const cv::Mat &mat)
{
    switch(mat.type())
    {
    case CV_8UC1:
    case CV_32FC1: return 1;
    case CV_8UC3:
    case CV_32FC3: return 3;
    case CV_8UC4:
    case CV_32FC4: return 4;
    default: return 0;
    }
}

int MainWindow::channelNum(const QImage &img)
{
    switch(img.format())
    {
    case QImage::Format_Indexed8:
    case QImage::Format_Grayscale8: return 1;
    case QImage::Format_RGB888: return 3;
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied: return 4;
    default: return 0;
    }
}

void MainWindow::showPicture(const QString &path)
{
    if (!path.isNull())
    {
        QPixmap pxm = QPixmap(path);
        pxm = pxm.scaled(ui->labelPictureMain->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->labelPictureMain->setPixmap(pxm);
        ui->labelPictureMain->setAlignment(Qt::AlignCenter);
    }
    else
        std::cout << "ERROR: path is null." << std::endl;
}

void MainWindow::showPicture(const cv::Mat &mat)
{
    if (!mat.empty())
    {
        QImage img = matToImg(mat);
        QPixmap pxm = QPixmap::fromImage(img);
        pxm = pxm.scaled(ui->labelPictureMain->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->labelPictureMain->setPixmap(pxm);
        ui->labelPictureMain->setAlignment(Qt::AlignCenter);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
}

void MainWindow::showChannel(const cv::Mat &mat)
{
    if (!mat.empty())
    {
        cv::Mat matCopy = mat.clone();
        std::vector<cv::Mat> chs;
        cv::split(matCopy, chs);
        switch(chs.size())
        {
        case 1:
            showMatrix("channel a", chs[3]); // a
            break;
        case 3:
            showMatrix("channel b", chs[0]); // b
            showMatrix("channel g", chs[1]); // g
            showMatrix("channel r", chs[2]); // r
            break;
        case 4:
            showMatrix("channel b", chs[0]); // b
            showMatrix("channel g", chs[1]); // g
            showMatrix("channel r", chs[2]); // r
            showMatrix("channel a", chs[3]); // a
            break;
        default:
            break;
        }
    }
    else
        std::cout << "ERROR: channel is null." << std::endl;
}

void MainWindow::showMatrix(const char *name, const cv::Mat &mat)
{
    if (!mat.empty())
    {
        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, mat);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
}

void MainWindow::cancelChanges()
{
    dstMat = srcMat; // roll back
    showPicture(srcMat);
    // std::cout << "undoLog: " << undoLog.size() << ", redoLog: " << redoLog.size() << std::endl;
}

void MainWindow::applyChanges()
{
    undoLog.push_back(srcMat.clone());
    if (undoLog.size() > LOGSIZE) { undoLog.pop_front(); }
    redoLog.clear();
    srcMat = dstMat; // update
    showPicture(dstMat);
    // std::cout << "undoLog: " << undoLog.size() << ", redoLog: " << redoLog.size() << std::endl;
}

void MainWindow::resetValuesE()
{
    ui->sliderEnhanceBr->setValue(0); ui->spinEnhanceBr->setValue(0);
    ui->sliderEnhanceCt->setValue(0); ui->spinEnhanceCt->setValue(0);
    ui->sliderEnhanceEx->setValue(0); ui->spinEnhanceEx->setValue(0);
    ui->sliderEnhanceHl->setValue(0); ui->spinEnhanceHl->setValue(0);
    ui->sliderEnhanceSd->setValue(0); ui->spinEnhanceSd->setValue(0);
    ui->sliderEnhanceSt->setValue(0); ui->spinEnhanceSt->setValue(0);
    ui->sliderEnhanceWm->setValue(0); ui->spinEnhanceWm->setValue(0);
    ui->sliderEnhanceTn->setValue(0); ui->spinEnhanceTn->setValue(0);
    ui->sliderEnhanceSp->setValue(0); ui->spinEnhanceSp->setValue(0);
    ui->sliderEnhanceSm->setValue(0); ui->spinEnhanceSm->setValue(0);
}

void MainWindow::resetValuesD()
{
    ui->sliderDegradeNg->setValue(0); ui->spinDegradeNg->setValue(0);
    ui->sliderDegradeNp->setValue(0); ui->spinDegradeNp->setValue(0);
    ui->sliderDegradeNi->setValue(0); ui->spinDegradeNi->setValue(0);
    ui->sliderDegradeNs->setValue(0); ui->spinDegradeNs->setValue(0);
    ui->sliderDegradeBd->setValue(0); ui->spinDegradeBd->setValue(0);
    ui->sliderDegradeBm->setValue(0); ui->spinDegradeBm->setValue(0);
    ui->sliderDegradeBz->setValue(0); ui->spinDegradeBz->setValue(0);
}


/**************** SLOT ****************/

/* top bar */

void MainWindow::onButtonOpenClicked()
{
    if (!srcMat.empty())
    {
        undoLog.push_back(srcMat.clone());
        if (undoLog.size() > LOGSIZE) { undoLog.pop_front(); }
        redoLog.clear();
    }
    openPath = QFileDialog::getOpenFileName(this, tr("open"), QDir::homePath(), tr("(*.jpg)\n(*.png)"));
    if (!openPath.isNull())
    {
        srcMat = cv::imread(openPath.toStdString());
    }
    else
        std::cout << "ERROR: open path is null." << std::endl;
    dstMat = srcMat; // initialize
    showPicture(openPath);
    std::cout << "origin path: " << openPath.toStdString() << std::endl;
    std::cout << "origin size: " << srcMat.size() << std::endl;
    std::cout << "INFO: open done." << std::endl;
}

void MainWindow::onButtonSaveClicked()
{
    savePath = QFileDialog::getSaveFileName(this, tr("save"), QDir::homePath(), tr("(*.jpg)\n(*.png)"));
    if (!savePath.isNull())
    {
        cv::imwrite(savePath.toStdString(), dstMat);
    }
    else
        std::cout << "ERROR: save path is null." << std::endl;
    std::cout << "INFO: save done." << std::endl;
}

void MainWindow::onButtonUndoClicked()
{
    if (!undoLog.empty())
    {
        redoLog.push_back(srcMat.clone());
        if (redoLog.size() > LOGSIZE) { redoLog.pop_front(); }
        srcMat = undoLog.back();
        undoLog.pop_back();
        // std::cout << "INFO: undo done." << std::endl;
    }
    else
        std::cout << "ERROR: undo log is empty." << std::endl;
    dstMat = srcMat; // initialize
    showPicture(dstMat);
    // std::cout << "undoLog: " << undoLog.size() << ", redoLog: " << redoLog.size() << std::endl;
}

void MainWindow::onButtonRedoClicked()
{
    if (!redoLog.empty())
    {
        undoLog.push_back(srcMat.clone());
        if (undoLog.size() > LOGSIZE) { undoLog.pop_front(); }
        srcMat = redoLog.back();
        redoLog.pop_back();
        // std::cout << "INFO: redo done." << std::endl;
    }
    else
        std::cout << "ERROR: redo log is empty." << std::endl;
    dstMat = srcMat; // initialize
    showPicture(dstMat);
    // std::cout << "undoLog: " << undoLog.size() << ", redoLog: " << redoLog.size() << std::endl;
}

/* page: size */

void MainWindow::onButtonCropClicked() {}

void MainWindow::onButtonCropSClicked() {}

void MainWindow::onButtonCropRClicked() {}

void MainWindow::onButtonRotateAClicked()
{
    if (!dstMat.empty())
    {
        dstMat = rotate(dstMat, 1);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: rotate anticlock." << std::endl;
}

void MainWindow::onButtonRotateCClicked()
{
    if (!dstMat.empty())
    {
        dstMat = rotate(dstMat, 0);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: rotate clockwise." << std::endl;
}

void MainWindow::onButtonMirrorHClicked()
{
    if (!dstMat.empty())
    {
        dstMat = mirror(dstMat, 1);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: mirror horizontal." << std::endl;
}

void MainWindow::onButtonMirrorVClicked()
{
    if (!dstMat.empty())
    {
        dstMat = mirror(dstMat, 0);
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: mirror vertical." << std::endl;
}

void MainWindow::onButtonCancelGClicked()
{
    cancelChanges();
    // std::cout << "INFO: cancel size done." << std::endl;
}

void MainWindow::onButtonApplyGClicked()
{
    applyChanges();
    // std::cout << "INFO: apply size done." << std::endl;
}

/* page: enhance */

void MainWindow::onSliderEnhanceValuesChanged()
{
    if (!srcMat.empty())
    {
        dstMat = srcMat;
        dstMat = brightness(dstMat, ui->sliderEnhanceBr->value());
        dstMat = contrast(dstMat, ui->sliderEnhanceCt->value());
        dstMat = exposure(dstMat, ui->sliderEnhanceEx->value());
        dstMat = highlight(dstMat, ui->sliderEnhanceHl->value());
        dstMat = shadow(dstMat, ui->sliderEnhanceSd->value());
        dstMat = saturation(dstMat, ui->sliderEnhanceSt->value());
        dstMat = warmth(dstMat, ui->sliderEnhanceWm->value());
        dstMat = tint(dstMat, ui->sliderEnhanceTn->value());
        dstMat = sharpness(dstMat, ui->sliderEnhanceSp->value());
        dstMat = smoothness(dstMat, ui->sliderEnhanceSm->value());
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: enhance slider." << std::endl;
}

void MainWindow::onSpinEnhanceValuesChanged()
{
    if (!srcMat.empty())
    {
        dstMat = srcMat;
        dstMat = brightness(dstMat, ui->spinEnhanceBr->value());
        dstMat = contrast(dstMat, ui->spinEnhanceCt->value());
        dstMat = exposure(dstMat, ui->spinEnhanceEx->value());
        dstMat = highlight(dstMat, ui->spinEnhanceHl->value());
        dstMat = shadow(dstMat, ui->spinEnhanceSd->value());
        dstMat = saturation(dstMat, ui->spinEnhanceSt->value());
        dstMat = warmth(dstMat, ui->spinEnhanceWm->value());
        dstMat = tint(dstMat, ui->spinEnhanceTn->value());
        dstMat = sharpness(dstMat, ui->spinEnhanceSp->value());
        dstMat = smoothness(dstMat, ui->spinEnhanceSm->value());
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: enhance spin." << std::endl;
}

void MainWindow::onButtonCancelEClicked()
{
    cancelChanges();
    resetValuesE();
    // std::cout << "INFO: cancel enhance done." << std::endl;
}

void MainWindow::onButtonApplyEClicked()
{
    applyChanges();
    resetValuesE();
    // std::cout << "INFO: apply enhance done." << std::endl;
}

/* page: degrade */

void MainWindow::onSliderDegradeValuesChanged()
{
    if (!srcMat.empty())
    {
        dstMat = srcMat;
        dstMat = noiseGaussian(dstMat, ui->sliderDegradeNg->value());
        dstMat = noisePoisson(dstMat, ui->sliderDegradeNp->value());
        dstMat = noiseImpulse(dstMat, ui->sliderDegradeNi->value());
        dstMat = noiseSpeckle(dstMat, ui->sliderDegradeNs->value());
        dstMat = blurDefocus(dstMat, ui->sliderDegradeBd->value());
        dstMat = blurMotion(dstMat, ui->sliderDegradeBm->value());
        dstMat = blurZoom(dstMat, ui->sliderDegradeBz->value());
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: degrade slider." << std::endl;
}

void MainWindow::onSpinDegradeValuesChanged()
{
    if (!srcMat.empty())
    {
        dstMat = srcMat;
        dstMat = noiseGaussian(dstMat, ui->spinDegradeNg->value());
        dstMat = noisePoisson(dstMat, ui->spinDegradeNp->value());
        dstMat = noiseImpulse(dstMat, ui->spinDegradeNi->value());
        dstMat = noiseSpeckle(dstMat, ui->spinDegradeNs->value());
        dstMat = blurDefocus(dstMat, ui->spinDegradeBd->value());
        dstMat = blurMotion(dstMat, ui->spinDegradeBm->value());
        dstMat = blurZoom(dstMat, ui->spinDegradeBz->value());
    }
    else
        std::cout << "ERROR: matrix is null." << std::endl;
    showPicture(dstMat);
    // std::cout << "INFO: degrade spin." << std::endl;
}

void MainWindow::onButtonCancelDClicked()
{
    cancelChanges();
    resetValuesD();
    // std::cout << "INFO: cancel degrade done." << std::endl;
}

void MainWindow::onButtonApplyDClicked()
{
    applyChanges();
    resetValuesD();
    // std::cout << "INFO: apply degrade done." << std::endl;
}

/* page: restore */

