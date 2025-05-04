#include "FFT.h"
#include <ctime>

int main()
{
	int col = 1000, row = 1000;
	int aligned_col = calc_aligned_col(col,aligned_size);

	float* data = (float*)_aligned_malloc(aligned_col * row * 2 * sizeof(float), aligned_size);

	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			data[i * aligned_col * 2 + j * 2] = 1;
			data[i * aligned_col * 2 + j * 2 + 1] = 0;
		}
	}
	int st = clock();
	FFT2D(data, row, col);
	IFFT2D(data, row, col);
	printf("Time: %d\n", clock() - st);

	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			printf("%f ", data[i * aligned_col * 2 + j * 2]);
		}
		printf("\n");
	}


	return 0;
}