#include <iostream>
#include <ctime>
#include "simplefft.h"
using namespace std;

#define M_PI 3.14159265358979323846
#define DOUBLE_PI 6.28318530717958647692

int main() {
    // 测试
	const size_t testsize = 1e6;
	vector<float> input;
	for (size_t i = 0; i < testsize; i++) {
		input.push_back(i);
	}
	for (size_t i = 0; i < testsize; i++) {
		input.push_back(0);
	}

	clock_t start, end;
	start = clock();
	vector<float> output = hybridFFT(input);
	vector<float> output2 = hybridIFFT(output);
	end = clock();
	cout << "hybridFFT: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

	for (size_t i = 0; i < testsize; i++) {
		if (abs(input[i] - output2[i]) > 1) {
			cout << "hybridFFT error!" << endl;
			break;
		}
	}

	return 0;
}