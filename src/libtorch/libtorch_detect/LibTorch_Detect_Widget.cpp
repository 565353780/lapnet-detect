#include "LibTorch_Detect_Widget.h"
#include "ui_LibTorch_Detect_Widget.h"

LibTorch_Detect_Widget::LibTorch_Detect_Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::LibTorch_Detect_Widget)
{
    ui->setupUi(this);

    libtorch_detector_ = new LibTorch_Detector();
}

LibTorch_Detect_Widget::~LibTorch_Detect_Widget()
{
    delete ui;
}

void LibTorch_Detect_Widget::on_Btn_Start_clicked(void)
{
    libtorch_detector_->detect();
}
