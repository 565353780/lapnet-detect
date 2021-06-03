#ifndef LIBTORCH_DETECTOR_H
#define LIBTORCH_DETECTOR_H
#include <torch/torch.h>
#include <iostream>

using namespace std;

class LibTorch_Detector
{
public:
    LibTorch_Detector();
    ~LibTorch_Detector();

    void detect();
};

#endif // LIBTORCH_DETECTOR_H
