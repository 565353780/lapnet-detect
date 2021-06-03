#include "LibTorch_Detector.h"

LibTorch_Detector::LibTorch_Detector()
{

}

LibTorch_Detector::~LibTorch_Detector()
{

}

void LibTorch_Detector::detect()
{
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
}
