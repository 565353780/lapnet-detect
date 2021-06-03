#ifndef LIBTORCH_DETECT_WIDGET_H
#define LIBTORCH_DETECT_WIDGET_H

#include <QWidget>
#include <QObject>
#include <QDebug>

#include "LibTorch_Detector.h"

namespace Ui {
class LibTorch_Detect_Widget;
}

class LibTorch_Detect_Widget : public QWidget
{
    Q_OBJECT

public:
    explicit LibTorch_Detect_Widget(QWidget *parent = nullptr);
    ~LibTorch_Detect_Widget();

public Q_SLOTS:
    void on_Btn_Start_clicked(void);

private:
    Ui::LibTorch_Detect_Widget *ui;

    LibTorch_Detector* libtorch_detector_;
};

#endif // LIBTORCH_DETECT_WIDGET_H
