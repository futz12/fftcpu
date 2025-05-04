#pragma once

#include <immintrin.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>

#define M_PI 3.14159265358979323846

const int primeListsize = 10007;
static int aligned_size = 64;


// 交换两个浮点数
inline void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// 通用数字反转函数
inline int reverse_index(int i, int r, int m) {
    int reversed = 0;
    for (int j = 0; j < m; ++j) {
        reversed = reversed * r + (i % r);
        i /= r;
    }
    return reversed;
}

// 判断是不是3的幂
inline bool is_power_of_three(int n) {
    return n > 0 && 1162261467 % n == 0;
}

// 判断是不是5的幂
inline bool is_power_of_five(int n) {
    return n > 0 && 1220703125 % n == 0;
}

// 判断是不是2的幂
inline bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// 判断是不是4的幂
inline bool is_power_of_four(int n) {
    return n > 0 && (n & (n - 1)) == 0 && (n & 0x55555555) != 0;
}

__forceinline __m256 multiply_complex_8(__m256 a_bI, __m256 c_dI) {
    __m256 a_aI = _mm256_moveldup_ps(a_bI);
    __m256 b_bI = _mm256_movehdup_ps(a_bI);

	__m256 bd_bcI = _mm256_mul_ps(b_bI, _mm256_permute_ps(c_dI, 0b10'11'00'01));
	return _mm256_fmaddsub_ps(a_aI, c_dI, bd_bcI);
}

struct numberkit {
    bool not_prime[primeListsize];
    int root[primeListsize];
    std::vector<uint32_t> pri;

    std::vector<uint32_t> fftbase = { 2,3,5 }; // 已经实现了的基数

    void _euler() {
        for (int i = 2; i < primeListsize; ++i) {
            if (!not_prime[i]) {
                pri.push_back(i);
            }
            for (int j = 0; j < pri.size(); ++j) {
                uint32_t pri_j = pri[j];

                if (i * pri_j >= primeListsize) break;
                not_prime[i * pri_j] = true;
                if (i % pri_j == 0)
                    break;
            }
        }
    }

    void _root() {
        for (uint32_t i = 0; i < pri.size(); i++) {
            root[pri[i]] = findroot(pri[i], true);
        }
    }

    inline bool isprime(uint32_t n) {
        if (n < primeListsize)
            return !not_prime[n];

        uint32_t end = static_cast<uint32_t>(sqrt(n));
        for (uint32_t start = 2; start <= end; start++) {
            if (n % start == 0) return false;
        }
        return true;
    }

    inline std::vector<uint32_t> factor(uint32_t n, bool defactor_fftbase = false) {
        std::vector<uint32_t> result;
        if (1 == n) {
            result.push_back(1);
            return result;
        }
        if (!defactor_fftbase) {
            for (int i = 0; i < fftbase.size(); i++) {
                uint32_t base = 1;
                while (n % fftbase[i] == 0) {
                    n = n / fftbase[i];
                    base *= fftbase[i];
                }
                if (base != 1) {
                    result.push_back(base);
                }
                if (1 == n)
                    return result;
            }
        }

        // 先检验 primeList 内的质数
        for (int i = 0; i < pri.size(); i++) {
            uint32_t pri_i = pri[i];

            while (n % pri_i == 0) {
                result.push_back(pri_i);
                n = n / pri_i;
            }
            if (1 == n)
                return result;
        }
        int end = static_cast<int>(sqrt(n));
        for (int start = primeListsize; start <= end; start++) {
            while (n % start == 0) {
                result.push_back(start);
                n = n / start;
            }
            if (1 == n)
                return result;
        }
        // n is prime
        result.push_back(n);
        return result;
    }

    inline uint32_t findroot(uint32_t n, bool init = false) {
        if (!init && n < primeListsize)
            return root[n];

        if (n == 2)
            return 1;

        std::vector<uint32_t> factors = factor(n - 1, true);

        for (uint32_t primeRootCandidate = 2; primeRootCandidate <= n - 1; primeRootCandidate++) {
            bool isRoot = true;
            for (uint32_t i = 0; i < factors.size(); i++) {
                if (qpow(primeRootCandidate, (n - 1) / factors[i], n) == 1) {
                    isRoot = false;
                    break;
                }
            }
            if (isRoot)
                return primeRootCandidate;
        }
        std::abort();
    }

    inline uint64_t rqpow(uint64_t a, uint64_t k, uint64_t n) {
        if (k == 0) return 1;
        return qpow(a, (n - 2) * k, n);
    }

    static inline uint32_t next2(uint32_t n) {
        unsigned int p = 1;
        if (n && !(n & (n - 1)))
            return n;

        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    inline uint64_t qpow(uint64_t a, uint64_t k, uint64_t n) {
        uint64_t res = 1;
        while (k) {
            if (k & 1) res = res * a % n;
            a = a * a % n;
            k >>= 1;
        }
        return res;
    }

    numberkit() {
        memset(not_prime, 0, sizeof(not_prime));

        _euler();
        _root();
    }
} nk;

// 基2 DIT FFT（非递归）,无转置
void fft_dit_radix2(float* x, int n) {
    for (int m = 2; m <= n; m <<= 1) {          // m: 当前处理的FFT大小
        int mh = m >> 1;                        // 半长度
        int k = 0;

#ifdef __AVX2__
        for (; k + 3 < mh; k += 4) {           // 旋转因子索引
            // 计算旋转因子 W = e^(-2πik/m)
            float theta0 = -2.0f * M_PI * k / m;
			float theta1 = -2.0f * M_PI * (k + 1) / m;
			float theta2 = -2.0f * M_PI * (k + 2) / m;
			float theta3 = -2.0f * M_PI * (k + 3) / m;

            __m128 w03_r, w03_i;

            w03_i = _mm_sincos_ps(&w03_r, _mm_setr_ps(theta0, theta1, theta2, theta3));

            __m256 w03 = _mm256_setr_m128(_mm_unpacklo_ps(w03_r, w03_i), _mm_unpackhi_ps(w03_r, w03_i));

            for (int j = 0; j < n; j += m) {
                int idx03_1 = 2 * (j + k);       // 蝶形输入1
                int idx03_2 = 2 * (j + k + mh);   // 蝶形输入2


                // 读取数据
				__m256 a03 = _mm256_loadu_ps(x + idx03_1);
				__m256 b03 = _mm256_loadu_ps(x + idx03_2);

                // 蝶形运算
                __m256 vt03 = multiply_complex_8(w03, b03);
				__m256 x03_1 = _mm256_add_ps(a03, vt03);
				__m256 x03_2 = _mm256_sub_ps(a03, vt03);

				_mm256_storeu_ps(x + idx03_1, x03_1);
				_mm256_storeu_ps(x + idx03_2, x03_2);
            }
        }
#endif

        for (; k < mh; ++k) {           // 旋转因子索引
            // 计算旋转因子 W = e^(-2πik/m)
            float theta = -2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                int idx1 = 2 * (j + k);       // 蝶形输入1
                int idx2 = 2 * (j + k + mh);   // 蝶形输入2

                // 读取数据
                float a_r = x[idx1], a_i = x[idx1 + 1];
                float b_r = x[idx2], b_i = x[idx2 + 1];

                // 计算旋转后的b: b * W
                float t_r = w_r * b_r - w_i * b_i;
                float t_i = w_r * b_i + w_i * b_r;

                // 蝶形运算
                x[idx1] = a_r + t_r;     // a + b*W
                x[idx1 + 1] = a_i + t_i;
                x[idx2] = a_r - t_r;     // a - b*W
                x[idx2 + 1] = a_i - t_i;
            }
        }
    }
}

// 基2 DIF FFT（非递归）,无转置
void fft_dif_radix2(float* x, int n) {
    // 1. 蝶形运算（Sande-Tukey）
    for (int m = n; m >= 2; m >>= 1) {           // m: 当前处理的FFT大小
        int mh = m >> 1;                         // 半长度
        int k = 0;

#ifdef __AVX2__
        for (; k + 3 < mh; k += 4) {           // 旋转因子索引
            // 计算旋转因子 W = e^(-2πik/m)
            float theta0 = -2.0f * M_PI * k / m;
			float theta1 = -2.0f * M_PI * (k + 1) / m;
			float theta2 = -2.0f * M_PI * (k + 2) / m;
			float theta3 = -2.0f * M_PI * (k + 3) / m;

			__m128 w03_r, w03_i;

            w03_i = _mm_sincos_ps(&w03_r,_mm_setr_ps(theta0,theta1,theta2,theta3));

            __m256 w03 = _mm256_setr_m128(_mm_unpacklo_ps(w03_r, w03_i), _mm_unpackhi_ps(w03_r, w03_i));

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                int idx0_1 = 2 * (j + k);       // 蝶形输入1
                int idx0_2 = 2 * (j + k + mh);   // 蝶形输入2

                // 读取数据
				__m256 a03 = _mm256_loadu_ps(x + idx0_1);
				__m256 b03 = _mm256_loadu_ps(x + idx0_2);

                // 蝶形运算
				__m256 s03 = _mm256_add_ps(a03, b03);
				__m256 d03 = _mm256_sub_ps(a03, b03);
                __m256 vt03 = multiply_complex_8(w03, d03);

				_mm256_store_ps(x + idx0_1, s03);
				_mm256_store_ps(x + idx0_2, vt03);
            }
        }
#endif

        for (; k < mh; ++k) {           // 旋转因子索引
            // 计算旋转因子 W = e^(-2πik/m)
            float theta = -2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                int idx1 = 2 * (j + k);       // 蝶形输入1
                int idx2 = 2 * (j + k + mh);   // 蝶形输入2

                // 读取数据
                float a_r = x[idx1], a_i = x[idx1 + 1];
                float b_r = x[idx2], b_i = x[idx2 + 1];

                // 计算sum和diff
                float sum_r = a_r + b_r;
                float sum_i = a_i + b_i;
                float diff_r = a_r - b_r;
                float diff_i = a_i - b_i;

                // 更新结果
                x[idx1] = sum_r;        // sum直接存储
                x[idx1 + 1] = sum_i;
                x[idx2] = diff_r * w_r - diff_i * w_i;  // diff乘以W后存储
                x[idx2 + 1] = diff_r * w_i + diff_i * w_r;
            }
        }
    }
}

__forceinline void ifft_dit_radix2(float* x, int n) {
    // 取共轭
	for (int i = 0; i < n; ++i) {
		x[2 * i + 1] = -x[2 * i + 1];
	}
	fft_dit_radix2(x, n);
	// 取共轭
	for (int i = 0; i < n; ++i) {
		x[2 * i + 1] = -x[2 * i + 1];
	}
}

// 基3 DIT FFT（非递归）
void fft_dit_radix3(float* x, int n) {
    // 2. 蝶形运算（基3 Cooley-Tukey）
    for (int m = 3; m <= n; m *= 3) {         // m: 当前处理的FFT大小
        int block_size = m / 3;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m} 和 W^2
            float theta = -2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);
            float w2_r = cosf(2 * theta), w2_i = sinf(2 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 三个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);

                float a0r = x[idx0], a0i = x[idx0 + 1];
                float a1r = x[idx1] * w_r - x[idx1 + 1] * w_i, a1i = x[idx1] * w_i + x[idx1 + 1] * w_r;
                float a2r = x[idx2] * w2_r - x[idx2 + 1] * w2_i, a2i = x[idx2] * w2_i + x[idx2 + 1] * w2_r;

                // 基3蝴蝶运算
                float t0r = a0r + a1r + a2r;
                float t0i = a0i + a1i + a2i;

                float w1r = cosf(-2 * M_PI / 3);
                float w1i = sinf(-2 * M_PI / 3);
                float w2r = cosf(-4 * M_PI / 3);
                float w2i = sinf(-4 * M_PI / 3);

                float t1r = a0r + (a1r * w1r - a1i * w1i) + (a2r * w2r - a2i * w2i);
                float t1i = a0i + (a1r * w1i + a1i * w1r) + (a2r * w2i + a2i * w2r);

                float t2r = a0r + (a1r * w2r - a1i * w2i) + (a2r * w1r - a2i * w1i);
                float t2i = a0i + (a1r * w2i + a1i * w2r) + (a2r * w1i + a2i * w1r);

                x[idx0] = t0r; x[idx0 + 1] = t0i;
                x[idx1] = t1r; x[idx1 + 1] = t1i;
                x[idx2] = t2r; x[idx2 + 1] = t2i;
            }
        }
    }
}

// 基3 DIT FFT（非递归）
void ifft_dit_radix3(float* x, int n) {
    // 2. 蝶形运算（基3 Cooley-Tukey）
    for (int m = 3; m <= n; m *= 3) {         // m: 当前处理的FFT大小
        int block_size = m / 3;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m} 和 W^2
            float theta = 2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);
            float w2_r = cosf(2 * theta), w2_i = sinf(2 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 三个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);

                float a0r = x[idx0], a0i = x[idx0 + 1];
                float a1r = x[idx1] * w_r - x[idx1 + 1] * w_i, a1i = x[idx1] * w_i + x[idx1 + 1] * w_r;
                float a2r = x[idx2] * w2_r - x[idx2 + 1] * w2_i, a2i = x[idx2] * w2_i + x[idx2 + 1] * w2_r;

                // 基3蝴蝶运算
                float t0r = a0r + a1r + a2r;
                float t0i = a0i + a1i + a2i;

                float w1r = cosf(2 * M_PI / 3);
                float w1i = sinf(2 * M_PI / 3);
                float w2r = cosf(4 * M_PI / 3);
                float w2i = sinf(4 * M_PI / 3);

                float t1r = a0r + (a1r * w1r - a1i * w1i) + (a2r * w2r - a2i * w2i);
                float t1i = a0i + (a1r * w1i + a1i * w1r) + (a2r * w2i + a2i * w2r);

                float t2r = a0r + (a1r * w2r - a1i * w2i) + (a2r * w1r - a2i * w1i);
                float t2i = a0i + (a1r * w2i + a1i * w2r) + (a2r * w1i + a2i * w1r);

                x[idx0] = t0r; x[idx0 + 1] = t0i;
                x[idx1] = t1r; x[idx1 + 1] = t1i;
                x[idx2] = t2r; x[idx2 + 1] = t2i;
            }
        }
    }
}


// 基4 DIT FFT（非递归）
void fft_dit_radix4(float* x, int n) {
    // 2. 蝶形运算（基4 Cooley-Tukey）
    for (int m = 4; m <= n; m *= 4) {         // m: 当前处理的FFT大小
        int block_size = m / 4;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m}, W^2, W^3
            float theta = -2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);
            float w2_r = cosf(2 * theta), w2_i = sinf(2 * theta);
            float w3_r = cosf(3 * theta), w3_i = sinf(3 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 四个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);
                int idx3 = 2 * (j + k + 3 * block_size);

                // 读取原始数据
                float a_r = x[idx0], a_i = x[idx0 + 1];
                float b_r = x[idx1], b_i = x[idx1 + 1];
                float c_r = x[idx2], c_i = x[idx2 + 1];
                float d_r = x[idx3], d_i = x[idx3 + 1];

                // 计算旋转后的b, c, d
                float b_rot_r = w_r * b_r - w_i * b_i;
                float b_rot_i = w_r * b_i + w_i * b_r;
                float c_rot_r = w2_r * c_r - w2_i * c_i;
                float c_rot_i = w2_r * c_i + w2_i * c_r;
                float d_rot_r = w3_r * d_r - w3_i * d_i;
                float d_rot_i = w3_r * d_i + w3_i * d_r;

                // 基4蝶形运算的线性组合
                x[idx0] = a_r + b_rot_r + c_rot_r + d_rot_r;  // X0
                x[idx0 + 1] = a_i + b_rot_i + c_rot_i + d_rot_i;
                x[idx1] = a_r + b_rot_i - c_rot_r - d_rot_i;  // X1
                x[idx1 + 1] = a_i - b_rot_r - c_rot_i + d_rot_r;
                x[idx2] = a_r - b_rot_r + c_rot_r - d_rot_r;  // X2
                x[idx2 + 1] = a_i - b_rot_i + c_rot_i - d_rot_i;
                x[idx3] = a_r - b_rot_i - c_rot_r + d_rot_i;  // X3
                x[idx3 + 1] = a_i + b_rot_r - c_rot_i - d_rot_r;
            }
        }
    }
}

void fft_dif_radix4(float* x, int n) {
    // 1. 蝶形运算（基4 Sande-Tukey）
    for (int m = n; m >= 4; m /= 4) {        // m: 当前处理的FFT大小
        int block_size = m / 4;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m}, W^2, W^3
            float theta = -2.0f * M_PI * k / m;
            float w_r = cosf(theta), w_i = sinf(theta);
            float w2_r = cosf(2 * theta), w2_i = sinf(2 * theta);
            float w3_r = cosf(3 * theta), w3_i = sinf(3 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 四个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);
                int idx3 = 2 * (j + k + 3 * block_size);

                // 读取原始数据
                float a_r = x[idx0], a_i = x[idx0 + 1];
                float b_r = x[idx1], b_i = x[idx1 + 1];
                float c_r = x[idx2], c_i = x[idx2 + 1];
                float d_r = x[idx3], d_i = x[idx3 + 1];

                // 计算四路信号的和与组合
                float sum_r = a_r + b_r + c_r + d_r;
                float sum_i = a_i + b_i + c_i + d_i;
                float tmp1_r = a_r + b_i - c_r - d_i;
                float tmp1_i = a_i - b_r - c_i + d_r;
                float tmp2_r = a_r - b_r + c_r - d_r;
                float tmp2_i = a_i - b_i + c_i - d_i;
                float tmp3_r = a_r - b_i - c_r + d_i;
                float tmp3_i = a_i + b_r - c_i - d_r;

                // 应用旋转因子并更新结果
                x[idx0] = sum_r;                     // 直接存储和
                x[idx0 + 1] = sum_i;
                x[idx1] = tmp1_r * w_r - tmp1_i * w_i;  // 应用W
                x[idx1 + 1] = tmp1_r * w_i + tmp1_i * w_r;
                x[idx2] = tmp2_r * w2_r - tmp2_i * w2_i;  // 应用W^2
                x[idx2 + 1] = tmp2_r * w2_i + tmp2_i * w2_r;
                x[idx3] = tmp3_r * w3_r - tmp3_i * w3_i;  // 应用W^3
                x[idx3 + 1] = tmp3_r * w3_i + tmp3_i * w3_r;
            }
        }
    }
}

void ifft_dit_radix4(float* x, int n) {
	// 取共轭
	for (int i = 0; i < n; ++i) {
		x[2 * i + 1] = -x[2 * i + 1];
	}
	fft_dit_radix4(x, n);
	// 取共轭
	for (int i = 0; i < n; ++i) {
		x[2 * i + 1] = -x[2 * i + 1];
	}
}

// 基5 DIT FFT（非递归）
void fft_dit_radix5(float* x, int n) {
    // 2. 蝶形运算（基5 Cooley-Tukey）
    for (int m = 5; m <= n; m *= 5) {         // m: 当前处理的FFT大小
        int block_size = m / 5;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m}, W^2, W^3, W^4
            float theta = -2 * M_PI * k / m;
            float wr1 = cosf(theta);
            float wi1 = sinf(theta);
            float wr2 = cosf(2 * theta);
            float wi2 = sinf(2 * theta);
            float wr3 = cosf(3 * theta);
            float wi3 = sinf(3 * theta);
            float wr4 = cosf(4 * theta);
            float wi4 = sinf(4 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 五个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);
                int idx3 = 2 * (j + k + 3 * block_size);
                int idx4 = 2 * (j + k + 4 * block_size);

                float a0r = x[idx0], a0i = x[idx0 + 1];
                float a1r = x[idx1] * wr1 - x[idx1 + 1] * wi1, a1i = x[idx1] * wi1 + x[idx1 + 1] * wr1;
                float a2r = x[idx2] * wr2 - x[idx2 + 1] * wi2, a2i = x[idx2] * wi2 + x[idx2 + 1] * wr2;
                float a3r = x[idx3] * wr3 - x[idx3 + 1] * wi3, a3i = x[idx3] * wi3 + x[idx3 + 1] * wr3;
                float a4r = x[idx4] * wr4 - x[idx4 + 1] * wi4, a4i = x[idx4] * wi4 + x[idx4 + 1] * wr4;

                // 基5蝴蝶运算
                float t0r = a0r + a1r + a2r + a3r + a4r;
                float t0i = a0i + a1i + a2i + a3i + a4i;

                float w1r = cosf(-2 * M_PI / 5);
                float w1i = sinf(-2 * M_PI / 5);
                float w2r = cosf(-4 * M_PI / 5);
                float w2i = sinf(-4 * M_PI / 5);
                float w3r = cosf(-6 * M_PI / 5);
                float w3i = sinf(-6 * M_PI / 5);
                float w4r = cosf(-8 * M_PI / 5);
                float w4i = sinf(-8 * M_PI / 5);

                float t1r = a0r + (a1r * w1r - a1i * w1i) + (a2r * w2r - a2i * w2i) + (a3r * w3r - a3i * w3i) + (a4r * w4r - a4i * w4i);
                float t1i = a0i + (a1r * w1i + a1i * w1r) + (a2r * w2i + a2i * w2r) + (a3r * w3i + a3i * w3r) + (a4r * w4i + a4i * w4r);

                float t2r = a0r + (a1r * w2r - a1i * w2i) + (a2r * w4r - a2i * w4i) + (a3r * w1r - a3i * w1i) + (a4r * w3r - a4i * w3i);
                float t2i = a0i + (a1r * w2i + a1i * w2r) + (a2r * w4i + a2i * w4r) + (a3r * w1i + a3i * w1r) + (a4r * w3i + a4i * w3r);

                float t3r = a0r + (a1r * w3r - a1i * w3i) + (a2r * w1r - a2i * w1i) + (a3r * w4r - a3i * w4i) + (a4r * w2r - a4i * w2i);
                float t3i = a0i + (a1r * w3i + a1i * w3r) + (a2r * w1i + a2i * w1r) + (a3r * w4i + a3i * w4r) + (a4r * w2i + a4i * w2r);

                float t4r = a0r + (a1r * w4r - a1i * w4i) + (a2r * w3r - a2i * w3i) + (a3r * w2r - a3i * w2i) + (a4r * w1r - a4i * w1i);
                float t4i = a0i + (a1r * w4i + a1i * w4r) + (a2r * w3i + a2i * w3r) + (a3r * w2i + a3i * w2r) + (a4r * w1i + a4i * w1r);

                x[idx0] = t0r; x[idx0 + 1] = t0i;
                x[idx1] = t1r; x[idx1 + 1] = t1i;
                x[idx2] = t2r; x[idx2 + 1] = t2i;
                x[idx3] = t3r; x[idx3 + 1] = t3i;
                x[idx4] = t4r; x[idx4 + 1] = t4i;
            }
        }
    }
}

void ifft_dit_radix5(float* x, int n) {
    // 2. 蝶形运算（基5 Cooley-Tukey）
    for (int m = 5; m <= n; m *= 5) {         // m: 当前处理的FFT大小
        int block_size = m / 5;               // 每组蝶形运算的块大小
        for (int k = 0; k < block_size; ++k) { // 旋转因子索引
            // 计算旋转因子 W = e^{-2πik/m}, W^2, W^3, W^4
            float theta = 2 * M_PI * k / m;
            float wr1 = cosf(theta);
            float wi1 = sinf(theta);
            float wr2 = cosf(2 * theta);
            float wi2 = sinf(2 * theta);
            float wr3 = cosf(3 * theta);
            float wi3 = sinf(3 * theta);
            float wr4 = cosf(4 * theta);
            float wi4 = sinf(4 * theta);

            // 遍历每组蝶形
            for (int j = 0; j < n; j += m) {
                // 五个输入点的索引
                int idx0 = 2 * (j + k);
                int idx1 = 2 * (j + k + block_size);
                int idx2 = 2 * (j + k + 2 * block_size);
                int idx3 = 2 * (j + k + 3 * block_size);
                int idx4 = 2 * (j + k + 4 * block_size);

                float a0r = x[idx0], a0i = x[idx0 + 1];
                float a1r = x[idx1] * wr1 - x[idx1 + 1] * wi1, a1i = x[idx1] * wi1 + x[idx1 + 1] * wr1;
                float a2r = x[idx2] * wr2 - x[idx2 + 1] * wi2, a2i = x[idx2] * wi2 + x[idx2 + 1] * wr2;
                float a3r = x[idx3] * wr3 - x[idx3 + 1] * wi3, a3i = x[idx3] * wi3 + x[idx3 + 1] * wr3;
                float a4r = x[idx4] * wr4 - x[idx4 + 1] * wi4, a4i = x[idx4] * wi4 + x[idx4 + 1] * wr4;

                // 基5蝴蝶运算
                float t0r = a0r + a1r + a2r + a3r + a4r;
                float t0i = a0i + a1i + a2i + a3i + a4i;

                float w1r = cosf(2 * M_PI / 5);
                float w1i = sinf(2 * M_PI / 5);
                float w2r = cosf(4 * M_PI / 5);
                float w2i = sinf(4 * M_PI / 5);
                float w3r = cosf(6 * M_PI / 5);
                float w3i = sinf(6 * M_PI / 5);
                float w4r = cosf(8 * M_PI / 5);
                float w4i = sinf(8 * M_PI / 5);

                float t1r = a0r + (a1r * w1r - a1i * w1i) + (a2r * w2r - a2i * w2i) + (a3r * w3r - a3i * w3i) + (a4r * w4r - a4i * w4i);
                float t1i = a0i + (a1r * w1i + a1i * w1r) + (a2r * w2i + a2i * w2r) + (a3r * w3i + a3i * w3r) + (a4r * w4i + a4i * w4r);

                float t2r = a0r + (a1r * w2r - a1i * w2i) + (a2r * w4r - a2i * w4i) + (a3r * w1r - a3i * w1i) + (a4r * w3r - a4i * w3i);
                float t2i = a0i + (a1r * w2i + a1i * w2r) + (a2r * w4i + a2i * w4r) + (a3r * w1i + a3i * w1r) + (a4r * w3i + a4i * w3r);

                float t3r = a0r + (a1r * w3r - a1i * w3i) + (a2r * w1r - a2i * w1i) + (a3r * w4r - a3i * w4i) + (a4r * w2r - a4i * w2i);
                float t3i = a0i + (a1r * w3i + a1i * w3r) + (a2r * w1i + a2i * w1r) + (a3r * w4i + a3i * w4r) + (a4r * w2i + a4i * w2r);

                float t4r = a0r + (a1r * w4r - a1i * w4i) + (a2r * w3r - a2i * w3i) + (a3r * w2r - a3i * w2i) + (a4r * w1r - a4i * w1i);
                float t4i = a0i + (a1r * w4i + a1i * w4r) + (a2r * w3i + a2i * w3r) + (a3r * w2i + a3i * w2r) + (a4r * w1i + a4i * w1r);

                x[idx0] = t0r; x[idx0 + 1] = t0i;
                x[idx1] = t1r; x[idx1 + 1] = t1i;
                x[idx2] = t2r; x[idx2 + 1] = t2i;
                x[idx3] = t3r; x[idx3 + 1] = t3i;
                x[idx4] = t4r; x[idx4 + 1] = t4i;
            }
        }
    }
}

inline void reverse(float* x, int n, int r) {
    int m = 0;
    int len = 1;
	while (len < n) {
		len *= r;
		m++;
	}

	for (int i = 0; i < n; ++i) {
		int reversed = reverse_index(i, r, m);
		if (reversed > i) {
			swap(&x[2 * i], &x[2 * reversed]);
			swap(&x[2 * i + 1], &x[2 * reversed + 1]);
		}
	}
}

inline void raderFFT(float* x, int n) {
    float X0r = 0, X0i = 0;

    for (int i = 0; i < n * 2; i += 2)
    {
		X0r += x[i];
		X0i += x[i + 1];
    }

    uint32_t g = nk.findroot(n);
    uint32_t m = nk.next2(2 * n - 3);

	float* aq = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);
	float* bq = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);

	memset(aq, 0, 2 * m * sizeof(float));
	memset(bq, 0, 2 * m * sizeof(float));

	float* product = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);

	uint32_t* bqIndex = (uint32_t*)_aligned_malloc(n * sizeof(uint32_t), aligned_size);

	bqIndex[0] = 1;

	aq[0] = x[2]; // aq[0].r = x[1].r
	aq[1] = x[3]; // aq[0].i = x[1].i

	bq[0] = cosf(2 * M_PI / n); // bq[0].r = cos(2 * M_PI / n)
	bq[1] = -sinf(2 * M_PI / n); // bq[0].i = -sin(2 * M_PI / n)

    uint64_t exp = nk.qpow(g, 1, n);
    uint64_t expInverse = nk.rqpow(g, 1, n);
    uint64_t expInverseBase = expInverse;


    for (uint32_t index = 1; index <= n - 2; index++) {
        bqIndex[index] = expInverse;

		aq[(m - n + 1 + index) * 2] = x[exp * 2]; // aq[m - n + 1 + index].r = x[exp].r
		aq[(m - n + 1 + index) * 2 + 1] = x[exp * 2 + 1]; // aq[m - n + 1 + index].i = x[exp].i

        float angle = expInverse * 2 * M_PI / n;

		bq[index * 2] = cosf(angle); // bq[index].r = cos(angle)
		bq[index * 2 + 1] = -sinf(angle); // bq[index].i = -sin(angle)

		exp = (exp * g) % n;
		expInverse = (expInverse * expInverseBase) % n;
    }

    // pad
    if (m != n - 1) {
        for (uint32_t index = 0; index < m - n + 1; index++) {
			bq[(n - 1 + index) * 2] = bq[index * 2]; // bq[n - 1 + index].r = bq[index].r
			bq[(n - 1 + index) * 2 + 1] = bq[index * 2 + 1]; // bq[n - 1 + index].i = bq[index].i
        }
    }

    // 这里是没有做翻转的，所以后面要配合dif做逆变换
    if (is_power_of_four(m)) {
		fft_dif_radix4(aq, m);
		fft_dif_radix4(bq, m);
    }
    else {
		fft_dif_radix2(aq, m);
		fft_dif_radix2(bq, m);
    }

    for (uint32_t index = 0; index < m; index++) {
		product[index * 2] = aq[index * 2] * bq[index * 2] - aq[index * 2 + 1] * bq[index * 2 + 1]; // product[index].r = aq[index].r * bq[index].r - aq[index].i * bq[index].i
		product[index * 2 + 1] = aq[index * 2] * bq[index * 2 + 1] + aq[index * 2 + 1] * bq[index * 2]; // product[index].i = aq[index].r * bq[index].i + aq[index].i * bq[index].r
    }

	if (is_power_of_four(m)) {
		ifft_dit_radix4(product, m);
	}
	else {
		ifft_dit_radix2(product, m);
	}
	
    float d0r = x[0];
    float d0i = x[1];

	x[0] = X0r;
	x[1] = X0i;

	for (uint32_t index = 0; index < n - 1; index++) {
		x[bqIndex[index] * 2] = product[index * 2] / m + d0r; // x[bqIndex[index]].r = product[index].r / m + d0.r
		x[bqIndex[index] * 2 + 1] = product[index * 2 + 1] / m + d0i; // x[bqIndex[index]].i = product[index].i / m + d0.i
	}
	
	_aligned_free(aq);
	_aligned_free(bq);
	_aligned_free(product);
	_aligned_free(bqIndex);
}

inline void raderIFFT(float* x, int n) {
    float X0r = 0, X0i = 0;

    for (int i = 0; i < n * 2; i += 2)
    {
        X0r += x[i];
        X0i += x[i + 1];
    }

    uint32_t g = nk.findroot(n);
    uint32_t m = nk.next2(2 * n - 3);

    float* aq = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);
    float* bq = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);

    memset(aq, 0, 2 * m * sizeof(float));
    memset(bq, 0, 2 * m * sizeof(float));

    float* product = (float*)_aligned_malloc(2 * m * sizeof(float), aligned_size);

    uint32_t* bqIndex = (uint32_t*)_aligned_malloc(n * sizeof(uint32_t), aligned_size);

    bqIndex[0] = 1;

    aq[0] = x[2]; // aq[0].r = x[1].r
    aq[1] = x[3]; // aq[0].i = x[1].i

    bq[0] = cosf(2 * M_PI / n); // bq[0].r = cos(2 * M_PI / n)
    bq[1] = sinf(2 * M_PI / n); // bq[0].i = -sin(2 * M_PI / n)

    uint64_t exp = nk.qpow(g, 1, n);
    uint64_t expInverse = nk.rqpow(g, 1, n);
    uint64_t expInverseBase = expInverse;


    for (uint32_t index = 1; index <= n - 2; index++) {
        bqIndex[index] = expInverse;

        aq[(m - n + 1 + index) * 2] = x[exp * 2]; // aq[m - n + 1 + index].r = x[exp].r
        aq[(m - n + 1 + index) * 2 + 1] = x[exp * 2 + 1]; // aq[m - n + 1 + index].i = x[exp].i

        float angle = expInverse * 2 * M_PI / n;

        bq[index * 2] = cosf(angle); // bq[index].r = cos(angle)
        bq[index * 2 + 1] = sinf(angle); // bq[index].i = -sin(angle)

        exp = (exp * g) % n;
        expInverse = (expInverse * expInverseBase) % n;
    }

    // pad
    if (m != n - 1) {
        for (uint32_t index = 0; index < m - n + 1; index++) {
            bq[(n - 1 + index) * 2] = bq[index * 2]; // bq[n - 1 + index].r = bq[index].r
            bq[(n - 1 + index) * 2 + 1] = bq[index * 2 + 1]; // bq[n - 1 + index].i = bq[index].i
        }
    }

    // 这里是没有做翻转的，所以后面要配合dif做逆变换
    if (is_power_of_four(m)) {
        fft_dif_radix4(aq, m);
        fft_dif_radix4(bq, m);
    }
    else {
        fft_dif_radix2(aq, m);
        fft_dif_radix2(bq, m);
    }

    for (uint32_t index = 0; index < m; index++) {
        product[index * 2] = aq[index * 2] * bq[index * 2] - aq[index * 2 + 1] * bq[index * 2 + 1]; // product[index].r = aq[index].r * bq[index].r - aq[index].i * bq[index].i
        product[index * 2 + 1] = aq[index * 2] * bq[index * 2 + 1] + aq[index * 2 + 1] * bq[index * 2]; // product[index].i = aq[index].r * bq[index].i + aq[index].i * bq[index].r
    }

    if (is_power_of_four(m)) {
        ifft_dit_radix4(product, m);
    }
    else {
        ifft_dit_radix2(product, m);
    }

    float d0r = x[0];
    float d0i = x[1];

    x[0] = X0r / n;
    x[1] = X0i / n;

    for (uint32_t index = 0; index < n - 1; index++) {
        x[bqIndex[index] * 2] = (product[index * 2] / m + d0r) / n;
        x[bqIndex[index] * 2 + 1] = (product[index * 2 + 1] / m + d0i) / n;
    }

    _aligned_free(aq);
    _aligned_free(bq);
    _aligned_free(product);
    _aligned_free(bqIndex);
}

void hybridFFT(float* x, int n) {
	if (n == 1) {
		return;
	}
    if (is_power_of_four(n)) {
        reverse(x, n, 4);
        fft_dit_radix4(x, n);
        return;
    }
	else if (is_power_of_two(n)) {
		reverse(x, n, 2);
		fft_dit_radix2(x, n);
        return;
	}
	else if (is_power_of_three(n)) {
		reverse(x, n, 3);
		fft_dit_radix3(x, n);
		return;
	}
	else if (is_power_of_five(n)) {
		reverse(x, n, 5);
		fft_dit_radix5(x, n);
		return;
	}
	
	std::vector<uint32_t> factors = nk.factor(n);

	if (factors.size() == 1) {
		raderFFT(x, n);
        return;
	}
	
    // 生成N1和N2，使得 N=N1*N2
    uint32_t N1 = 1;
    for (size_t i = 0; i < factors.size() - 1; i++) { // 必须留一个质因子出来，即为N2必须不为1，不然会死循环
        N1 *= factors[i];
        if (N1 * N1 >= n) {
            break;
        }
    }
    uint32_t N2 = n / N1;

	float* row = (float*)_aligned_malloc(2 * N2 * sizeof(float), aligned_size);

	float* tmpX = (float*)_aligned_malloc(2 * N2 * N1 * sizeof(float), aligned_size);

    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t n2 = 0; n2 < N2; n2++) {
			row[n2 * 2] = x[(n1 + n2 * N1) * 2];
			row[n2 * 2 + 1] = x[(n1 + n2 * N1) * 2 + 1];
        }

		hybridFFT(row, N2);

        for (uint32_t n2 = 0; n2 < N2; n2++) {
			float angle = -2 * M_PI * n1 * n2 / n;
			float w_r = cosf(angle);
			float w_i = sinf(angle);

			tmpX[(n1 * N2 + n2) * 2] = row[n2 * 2] * w_r - row[n2 * 2 + 1] * w_i;
			tmpX[(n1 * N2 + n2) * 2 + 1] = row[n2 * 2] * w_i + row[n2 * 2 + 1] * w_r;
        }
    }

	float* col = (float*)_aligned_malloc(2 * N1 * sizeof(float), aligned_size);

    for (uint32_t n2 = 0; n2 < N2; n2++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
			col[n1 * 2] = tmpX[(n1 * N2 + n2) * 2];
			col[n1 * 2 + 1] = tmpX[(n1 * N2 + n2) * 2 + 1];
        }

		hybridFFT(col, N1);

        for (uint32_t n1 = 0; n1 < N1; n1++) {
            x[(n1 * N2 + n2) * 2] = col[n1 * 2];
            x[(n1 * N2 + n2) * 2 + 1] = col[n1 * 2 + 1];
        }
    }

	_aligned_free(row);
	_aligned_free(col);
	_aligned_free(tmpX);
}

void hybridIFFT(float* x, int n) {
    if (n == 1) {
        return;
    }
    if (is_power_of_four(n)) {
        reverse(x, n, 4);
        ifft_dit_radix4(x, n);
        for (int i = 0; i < n; ++i) {
            x[2 * i] /= n;
            x[2 * i + 1] /= n;
        }
        return;
    }
    else if (is_power_of_two(n)) {
        reverse(x, n, 2);
        ifft_dit_radix2(x, n);
        for (int i = 0; i < n; ++i) {
            x[2 * i] /= n;
            x[2 * i + 1] /= n;
        }
        return;
    }
    else if (is_power_of_three(n)) {
        reverse(x, n, 3);
        ifft_dit_radix3(x, n);
		for (int i = 0; i < n; ++i) {
			x[2 * i] /= n;
			x[2 * i + 1] /= n;
		}
        return;
    }
    else if (is_power_of_five(n)) {
        reverse(x, n, 5);
        ifft_dit_radix5(x, n);
		for (int i = 0; i < n; ++i) {
			x[2 * i] /= n;
			x[2 * i + 1] /= n;
		}
        return;
    }

    std::vector<uint32_t> factors = nk.factor(n);

    if (factors.size() == 1) {
        raderIFFT(x, n);
        return;
    }

    // 生成N1和N2，使得 N=N1*N2
    uint32_t N1 = 1;
    for (size_t i = 0; i < factors.size() - 1; i++) { // 必须留一个质因子出来，即为N2必须不为1，不然会死循环
        N1 *= factors[i];
        if (N1 * N1 >= n) {
            break;
        }
    }
    uint32_t N2 = n / N1;

    float* row = (float*)_aligned_malloc(2 * N2 * sizeof(float), aligned_size);

    float* tmpX = (float*)_aligned_malloc(2 * N2 * N1 * sizeof(float), aligned_size);

    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t n2 = 0; n2 < N2; n2++) {
            row[n2 * 2] = x[(n1 + n2 * N1) * 2];
            row[n2 * 2 + 1] = x[(n1 + n2 * N1) * 2 + 1];
        }

        hybridIFFT(row, N2);

        for (uint32_t n2 = 0; n2 < N2; n2++) {
            float angle = 2 * M_PI * n1 * n2 / n;
            float w_r = cosf(angle);
            float w_i = sinf(angle);

            tmpX[(n1 * N2 + n2) * 2] = (row[n2 * 2] * w_r - row[n2 * 2 + 1] * w_i);
            tmpX[(n1 * N2 + n2) * 2 + 1] = (row[n2 * 2] * w_i + row[n2 * 2 + 1] * w_r);
        }
    }

    float* col = (float*)_aligned_malloc(2 * N1 * sizeof(float), aligned_size);

    for (uint32_t n2 = 0; n2 < N2; n2++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            col[n1 * 2] = tmpX[(n1 * N2 + n2) * 2];
            col[n1 * 2 + 1] = tmpX[(n1 * N2 + n2) * 2 + 1];
        }

        hybridIFFT(col, N1);

        for (uint32_t n1 = 0; n1 < N1; n1++) {
            x[(n1 * N2 + n2) * 2] = col[n1 * 2];
            x[(n1 * N2 + n2) * 2 + 1] = col[n1 * 2 + 1];
        }
    }

    _aligned_free(row);
    _aligned_free(col);
    _aligned_free(tmpX);
}

inline int calc_aligned_col(int col, int aligned_bytes) {
    const int floats_per_complex = 2;
    const int floats_per_aligned = aligned_bytes / sizeof(float);

    // 计算需要的最小float数量（包含填充）
    int total_floats = (col * floats_per_complex + floats_per_aligned - 1) /
        floats_per_aligned * floats_per_aligned;

    return total_floats / floats_per_complex; // 返回对齐后的复数数量
}

void FFT2D(float* data, int rows, int col) {
    const int aligned_col = calc_aligned_col(col, aligned_size);
    const int stride = aligned_col * 2; // 每行的float数量（含填充）

    // 对齐的列处理缓冲区（存储复数）
    float* col_buf = (float*)_aligned_malloc(rows * 2 * sizeof(float), aligned_size);

    // 行方向FFT（处理有效数据区）
    for (int i = 0; i < rows; ++i) {
        hybridFFT(&data[i * stride], col); // 每个复数包含2个float
    }

    // 列方向FFT
    for (int j = 0; j < col; ++j) { // 只处理有效列
        // 提取列数据（考虑交错存储）
        for (int i = 0; i < rows; ++i) {
            const int offset = i * stride + j * 2;
            col_buf[i * 2] = data[offset];     // 实部
            col_buf[i * 2 + 1] = data[offset + 1]; // 虚部
        }

        hybridFFT(col_buf, rows); // 处理复数列

        // 写回数据
        for (int i = 0; i < rows; ++i) {
            const int offset = i * stride + j * 2;
            data[offset] = col_buf[i * 2];
            data[offset + 1] = col_buf[i * 2 + 1];
        }
    }

    _aligned_free(col_buf);
}

void IFFT2D(float* data, int rows, int col) {
    const int aligned_col = calc_aligned_col(col, aligned_size);
    const int stride = aligned_col * 2;

    float* col_buf = (float*)_aligned_malloc(rows * 2 * sizeof(float), aligned_size);

    // 列方向IFFT
    for (int j = 0; j < col; ++j) {
        for (int i = 0; i < rows; ++i) {
            const int offset = i * stride + j * 2;
            col_buf[i * 2] = data[offset];
            col_buf[i * 2 + 1] = data[offset + 1];
        }
        hybridIFFT(col_buf, rows);
        for (int i = 0; i < rows; ++i) {
            const int offset = i * stride + j * 2;
            data[offset] = col_buf[i * 2];
            data[offset + 1] = col_buf[i * 2 + 1];
        }
    }

    for (int i = 0; i < rows; ++i) {
        hybridIFFT(&data[i * stride], col);
    }

    _aligned_free(col_buf);
}