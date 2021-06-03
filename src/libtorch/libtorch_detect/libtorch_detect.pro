QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

QMAKE_CXXFLAGS += -DGLIBCXX_USE_CXX11_ABI=0
CONFIG += no_keywords

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    LibTorch_Detect_Widget.cpp \
    LibTorch_Detector.cpp \
    main.cpp \
    MainWidget.cpp

HEADERS += \
    LibTorch_Detect_Widget.h \
    LibTorch_Detector.h \
    MainWidget.h

INCLUDEPATH += \
    /home/abaci/LibTorch/libtorch-shared-with-deps-1.7.0+cu110/libtorch/include \
    /home/abaci/LibTorch/libtorch-shared-with-deps-1.7.0+cu110/libtorch/include/torch/csrc/api/include

LIBS += \
    -L/home/abaci/LibTorch/libtorch-shared-with-deps-1.7.0+cu110/libtorch/lib \
    -ltorch \
    -lc10

FORMS += \
    LibTorch_Detect_Widget.ui \
    MainWidget.ui

TRANSLATIONS += \
    libtorch_detect_zh_CN.ts

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
