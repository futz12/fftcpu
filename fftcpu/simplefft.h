//
// Created by Ice on 2024/11/14.
//

#ifndef SIMPLEFFT_H
#define SIMPLEFFT_H

#include <complex>
#include <vector>


std::vector<float> hybridFFT(const std::vector<float> &x);
std::vector<float> hybridIFFT(const std::vector<float>& x);

#endif //SIMPLEFFT_H
