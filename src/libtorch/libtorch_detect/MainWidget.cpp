#include "MainWidget.h"
#include "ui_MainWidget.h"

MainWidget::MainWidget(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::MainWidget)
{
    ui->setupUi(this);

    libtorch_detect_widget_ = new LibTorch_Detect_Widget(this);
    this->ui->VLayout->addWidget(libtorch_detect_widget_);
}

MainWidget::~MainWidget()
{
    delete ui;
}

