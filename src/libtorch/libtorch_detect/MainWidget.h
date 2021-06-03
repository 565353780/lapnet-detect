#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>

#include "LibTorch_Detect_Widget.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWidget; }
QT_END_NAMESPACE

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    MainWidget(QWidget *parent = nullptr);
    ~MainWidget();

private:
    Ui::MainWidget *ui;

    LibTorch_Detect_Widget* libtorch_detect_widget_;
};
#endif // MAINWIDGET_H
