//
// Created by Ice on 2024/11/14.
//

#include <complex>
#include <numeric>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

#include "simplefft.h"

const int primeListsize = 10007;

struct numberkit {
    bool not_prime[primeListsize]{};
	uint32_t root[primeListsize]{};
    std::vector<uint32_t> pri;

    void _euler() {
        for (int i = 2; i < primeListsize; ++i) {
            if (!not_prime[i]) {
                pri.push_back(i);
            }
            for (int pri_j: pri) {
                if (i * pri_j >= primeListsize) break;
                not_prime[i * pri_j] = true;
                if (i % pri_j == 0)
                    break;
            }
        }
    }

	void _root() {
		for (uint32_t i = 0; i < pri.size(); i++) {
			root[pri[i]] = findroot(pri[i],true);
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

    inline std::vector<uint32_t> factor(uint32_t n) {
        std::vector<uint32_t> result;
        if (1 == n) {
            result.emplace_back(1);
            return result;
        }
        // 先检验 primeList 内的质数
        for (uint32_t pri: pri) {
            while (n % pri == 0) {
                result.emplace_back(pri);
                n = n / pri;
            }
            if (1 == n)
                return result;
        }
        int end = static_cast<int>(sqrt(n));
        for (int start = primeListsize; start <= end; start++) {
            while (n % start == 0) {
                result.emplace_back(start);
                n = n / start;
            }
            if (1 == n)
                return result;
        }
        // n is prime
        result.emplace_back(n);
        return result;
    }

    inline uint32_t findroot(uint32_t n,bool init = false) {
		if (init)
			return root[n];

        std::vector<uint32_t> factors = nk.factor(n - 1);

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

    inline uint32_t rqpow(uint32_t a, uint32_t k, uint32_t n) {
        if (k == 0) return 1;
        return qpow(a, (n - 2) * k, n);
    }

    inline uint32_t next2(uint32_t n) {
        if (n < 0)
            return n;
        unsigned int p = 1;
        if (n && !(n & (n - 1)))
            return n;

        while (p < n) {
            p <<= 1;
        }
        return p;
    }

    inline uint32_t qpow(uint32_t a, uint32_t k, uint32_t n) {
        uint32_t res = 1;
        while (k) {
            if (k & 1) res = res * a % n;
            a = a * a % n;
            k >>= 1;
        }
        return res;
    }

    numberkit() {
        _euler();
		_root();
    }
} nk;


#define DOUBLE_PI 6.283185307179586476925286766559

void fft0(int n, int s, bool eo, float* cx, float* cy)
{
	int len = n;

    float* xr = cx, * xi = cx + len;
    float* yr = cy, * yi = cy + len;

    for (; n > 1; n /= 2, s *= 2, eo = !eo, std::swap(xr, yr), std::swap(xi, yi)) {
        

        const int m = n / 2;
        const double theta0 = 2 * M_PI / n;

        for (int p = 0; p < m; p++) {
             // const complex_t wp = complex_t(cos(p * theta0), -sin(p * theta0));
			 float wpr = cosf(p * theta0);
			 float wpi = -sinf(p * theta0);
             for (int q = 0; q < s; q++) {
                 // const complex_t a = x[q + s * (p + 0)];
				 float ar = xr[q + s * (p + 0)];
				 float ai = xi[q + s * (p + 0)];
                 
                 // const complex_t b = x[q + s * (p + m)];
				 float br = xr[q + s * (p + m)];
				 float bi = xi[q + s * (p + m)];

                 // y[q + s * (2 * p + 0)] = a + b;
				 yr[q + s * (2 * p + 0)] = ar + br;
				 yi[q + s * (2 * p + 0)] = ai + bi;

                 // y[q + s * (2 * p + 1)] = (a - b) * wp;
				 yr[q + s * (2 * p + 1)] = (ar - br) * wpr - (ai - bi) * wpi;
				 yi[q + s * (2 * p + 1)] = (ar - br) * wpi + (ai - bi) * wpr;
             }
            }
    }
    if (eo) 
        for (int q = 0; q < s; q++) {
			yr[q] = xr[q];
			yi[q] = xi[q];
        }

    return;
}


void StockhamFFT(float * data,int len) {
    float*y = new float[len * 2];
    fft0(len, 1, 0, data, y);
    delete[] y;
}

void StockhamIFFT(float* data, int len) {

    for (int p = len; p < len * 2; p++) data[p] = -data[p]; // conj
    float *y = new float[len * 2];
    fft0(len, 1, 0, data, y);
    delete[] y;
    for (int p = len; p < len * 2; p++) data[p] = -data[p]; // conj
}

void raderFFT(std::vector<float> &data) {
    size_t N = data.size() / 2;
    // float X0 = std::accumulate(data.cbegin(), data.cend(), complex_t(0.0, 0.0));

	float X0r = 0, X0i = 0;
	for (int i = 0; i < N; i++) {
		X0r += data[i];
		X0i += data[i + N];
	}

    uint32_t g = nk.findroot(N);

    uint32_t M = nk.next2(2 * N - 3);

    float* aq,* bq;

	aq = new float[M * 2];
	bq = new float[M * 2];

	memset(aq, 0, M * 2 * sizeof(float));
	memset(bq, 0, M * 2 * sizeof(float));

	float* aqr, * bqr, * aqi, * bqi;
	aqr = aq;
	bqr = bq;
	aqi = aq + M;
	bqi = bq + M;

	float* product = new float[M * 2];

    float* productr,* producti;
	productr = product;
	producti = product + M;

    uint32_t* aqIndex, *bqIndex;

	aqIndex = new uint32_t[N];
	bqIndex = new uint32_t[N];

    aqIndex[0] = 1;
    bqIndex[0] = 1;

    // aq.emplace_back(data[1]);
	aqr[0] = data[1];
	aqi[0] = data[1 + N];

    // bq.emplace_back(complex_t(cosf(DOUBLE_PI / N), -sinf(DOUBLE_PI / N)));
    bqr[0] = cosf(DOUBLE_PI / N);
    bqi[0] = - sinf(DOUBLE_PI / N);

    uint32_t exp = nk.qpow(g, static_cast<size_t>(1), N);
    uint32_t expInverse = nk.rqpow(g, static_cast<size_t>(1), N);
    uint32_t expInverseBase = expInverse;

    for (size_t index = 1; index <= N - 2; index++) {
        aqIndex[index] = exp;
        bqIndex[index] = expInverse;

        // aq.emplace_back(data[exp]);
		aqr[M - N + 1 + index] = data[exp];
		aqi[M - N + 1 + index] = data[exp + N];
        
        auto tmp = expInverse * DOUBLE_PI / N;
        // bq.emplace_back(complex_t(cosf(tmp), -sinf(tmp)));
		bqr[index] = cosf(tmp);
		bqi[index] = -sinf(tmp);

        exp = (exp * g) % N;
        expInverse = (expInverse * expInverseBase) % N;
    }

    // 补零
    
    if (M != N - 1) {
        // aq.insert(aq.begin() + 1, M - N + 1, complex_t(0.0, 0.0));
		// aqr.insert(aqr.begin() + 1, M - N + 1, 0);
		// aqi.insert(aqi.begin() + 1, M - N + 1, 0);
        for (size_t index = 0; index < M - N + 1; index++) {
            // bq.emplace_back(bq[index]);
			bqr[N - 1 + index] = bqr[index];
			bqi[N - 1 + index] = bqi[index];
        }
    }

    StockhamFFT(aq,M);
    StockhamFFT(bq,M);

    for (size_t index = 0; index <= M - 1; index++) {
        // product.emplace_back(faq[index] * fbq[index]);
		float faqr = aq[index];
		float faqi = aq[index + M];
		float fbqr = bq[index];
		float fbqi = bq[index + M];

        productr[index] = (faqr* fbqr - faqi * fbqi);
		producti[index] = (faqr * fbqi + faqi * fbqr);
    }

    StockhamIFFT(product,M);

    std::vector<float> result(N * 2, 0.0);
    result[0] = X0r;
	result[N] = X0i;

    for (size_t index = 0; index < N - 1; index++) {
        // result[bqIndex[index]] = inverseDFT[index] / static_cast<complex_t>(M) + data[0];
		result[bqIndex[index]] = (product[index] / static_cast<float>(M) + data[0]);
		result[bqIndex[index] + N] = (product[index + M] / static_cast<float>(M) + data[0 + N]);
    }

	delete[] aq;
	delete[] bq;
	delete[] aqIndex;
	delete[] bqIndex;
    delete[] product;

    return result;
}

std::vector<float> raderIFFT(const std::vector<float>& data) {
    size_t N = data.size() / 2;
    // float X0 = std::accumulate(data.cbegin(), data.cend(), complex_t(0.0, 0.0));

    float X0r = 0, X0i = 0;
    for (int i = 0; i < N; i++) {
        X0r += data[i];
        X0i += data[i + N];
    }

    uint32_t g = nk.findroot(N);

    uint32_t M = nk.next2(2 * N - 3);

    float* aq, * bq;

    aq = new float[M * 2];
    bq = new float[M * 2];

    memset(aq, 0, M * 2 * sizeof(float));
    memset(bq, 0, M * 2 * sizeof(float));

    float* aqr, * bqr, * aqi, * bqi;
    aqr = aq;
    bqr = bq;
    aqi = aq + M;
    bqi = bq + M;

    float* product = new float[M * 2];

    float* productr, * producti;
    productr = product;
    producti = product + M;

    uint32_t* aqIndex, * bqIndex;

    aqIndex = new uint32_t[N];
    bqIndex = new uint32_t[N];

    aqIndex[0] = 1;
    bqIndex[0] = 1;

    // aq.emplace_back(data[1]);
    aqr[0] = data[1];
    aqi[0] = data[1 + N];

    // bq.emplace_back(complex_t(cosf(DOUBLE_PI / N), -sinf(DOUBLE_PI / N)));
    bqr[0] = cosf(DOUBLE_PI / N);
    bqi[0] = sinf(DOUBLE_PI / N);

    uint32_t exp = nk.qpow(g, static_cast<size_t>(1), N);
    uint32_t expInverse = nk.rqpow(g, static_cast<size_t>(1), N);
    uint32_t expInverseBase = expInverse;

    for (size_t index = 1; index <= N - 2; index++) {
        aqIndex[index] = exp;
        bqIndex[index] = expInverse;

        // aq.emplace_back(data[exp]);
        aqr[M - N + 1 + index] = data[exp];
        aqi[M - N + 1 + index] = data[exp + N];

        auto tmp = expInverse * DOUBLE_PI / N;
        // bq.emplace_back(complex_t(cosf(tmp), -sinf(tmp)));
        bqr[index] = cosf(tmp);
        bqi[index] = sinf(tmp);

        exp = (exp * g) % N;
        expInverse = (expInverse * expInverseBase) % N;
    }

    // 补零
    if (M != N - 1) {
        // aq.insert(aq.begin() + 1, M - N + 1, complex_t(0.0, 0.0));
        // aqr.insert(aqr.begin() + 1, M - N + 1, 0);
        // aqi.insert(aqi.begin() + 1, M - N + 1, 0);
        for (size_t index = 0; index < M - N + 1; index++) {
            // bq.emplace_back(bq[index]);
            bqr[N - 1 + index] = bqr[index];
            bqi[N - 1 + index] = bqi[index];
        }
    }

    StockhamFFT(aq, M);
    StockhamFFT(bq, M);

    for (size_t index = 0; index <= M - 1; index++) {
        // product.emplace_back(faq[index] * fbq[index]);
        float faqr = aq[index];
        float faqi = aq[index + M];
        float fbqr = bq[index];
        float fbqi = bq[index + M];

        productr[index] = (faqr * fbqr - faqi * fbqi);
        producti[index] = (faqr * fbqi + faqi * fbqr);
    }

    StockhamIFFT(product,M);
    std::vector<float> result(N * 2, 0.0);

    result[0] = X0r / N;
    result[N] = X0i / N;

    for (size_t index = 0; index < N - 1; index++) {
        // result[bqIndex[index]] = inverseDFT[index] / static_cast<complex_t>(M) + data[0];
        result[bqIndex[index]] = (product[index] / static_cast<float>(M) + data[0]) / N;
        result[bqIndex[index] + N] = (product[index + M] / static_cast<float>(M) + data[0 + N]) / N;
    }
    
    delete[] aq;
    delete[] bq;
    delete[] aqIndex;
    delete[] bqIndex;
	delete[] product;

    return result;
}

template<typename T>
T nextPowerOf2(T n) {
    if (n < 0)
        return n;
    unsigned int p = 1;
    if (n && !(n & (n - 1)))
        return n;

    while (p < n) {
        p <<= 1;
    }
    return p;
}

void hybridFFT(std::vector<float> &data) {
    size_t N = data.size() / 2;
    if (N == 1 || N == 2) {
        StockhamFFT(data.data(),data.size());
    }

    // 如果N是质数
    std::vector<uint32_t> factors = nk.factor(N);
    if (factors.size() == 1) {
        return raderFFT(data);
    }

    // 生成N1和N2，使得 N=N1*N2
    size_t N1 = factors[0], N2 = N / N1;

    // complex_t *X;
    // X = new complex_t [N1*N2];
	float* Xr = new float[N1 * N2];
	float* Xi = new float[N1 * N2];
    
    /*for (size_t n1 = 0; n1 < N1; n1++)
        for (size_t n2 = 0; n2 < N2; n2++)
            X[n1*N2 + n2] = data[N1 * n2 + n1];
    */

	for (int n1 = 0; n1 < N1; n1++) {
		for (int n2 = 0; n2 < N2; n2++) {
			Xr[n1 * N2 + n2] = data[N1 * n2 + n1];
			Xi[n1 * N2 + n2] = data[N1 * n2 + n1 + N];
		}
	}

    for (size_t n1 = 0; n1 < N1; n1++) {
        std::vector<float> row;
        row.resize(N2 * 2);
        for (size_t i = 0; i < N2; i++) {
            // row.emplace_back(X[n1 * N2 + i]);
			row[i] = Xr[n1 * N2 + i];
			row[i + N2] = Xi[n1 * N2 + i];
        }
        std::vector<float> tmp;

        if (nextPowerOf2(row.size()) != row.size())
            tmp = hybridFFT(row);
        else
            tmp = StockhamFFT(row);

        for (size_t n2 = 0; n2 < N2; n2++) {
            // X[n1 * N2 + n2] = tmp[n2] * complex_t(cosf(DOUBLE_PI * n1 * n2 / N), -sinf(DOUBLE_PI * n1 * n2 / N));
			float tmpr = tmp[n2];
			float tmpi = tmp[n2 + N2];
			float wr = cosf(DOUBLE_PI * n1 * n2 / N);
			float wi = -sinf(DOUBLE_PI * n1 * n2 / N);

			Xr[n1 * N2 + n2] = tmpr * wr - tmpi * wi;
			Xi[n1 * N2 + n2] = tmpr * wi + tmpi * wr;
        }
    }

    for (size_t n2 = 0; n2 < N2; n2++) {
        std::vector<float> col;
        col.resize(N1 * 2);
        for (size_t n1 = 0; n1 < N1; n1++)
        {
            // col.emplace_back(X[n1 * N2 + n2]);
            col[n1] = Xr[n1 * N2 + n2];
            col[n1 + N1] = Xi[n1 * N2 + n2];
        }

        std::vector<float> tmp;

        if (nextPowerOf2(col.size()) != col.size())
            tmp = hybridFFT(col);
        else
            tmp = StockhamFFT(col);

        for (size_t n1 = 0; n1 < N1; n1++) {
            // X[n1 * N2 + n2] = tmp[n1];
			Xr[n1 * N2 + n2] = tmp[n1];
			Xi[n1 * N2 + n2] = tmp[n1 + N1];
        }
    }

    std::vector<float> result(data.size(), 0.0);
    for (size_t n1 = 0; n1 < N1; n1++)
        for (size_t n2 = 0; n2 < N2; n2++) {
            // result[N2 * n1 + n2] = X[n1 * N2 + n2];
			result[N2 * n1 + n2] = Xr[n1 * N2 + n2];
			result[N2 * n1 + n2 + N] = Xi[n1 * N2 + n2];
        }

    delete [] Xr;
	delete[] Xi;

    return result;
}

std::vector<float> hybridIFFT(const std::vector<float>& data) {
    size_t N = data.size() / 2;
    if (N == 1 || N == 2) {
        std::vector<float> result = StockhamIFFT(data);
        for (int i = 0; i < 2 * N; i++)
            result[i] /= N;
        return result;
    }

    // 如果N是质数
    std::vector<uint32_t> factors = nk.factor(N);
    if (factors.size() == 1) {
        return raderIFFT(data);
    }

    // 生成N1和N2，使得 N=N1*N2
    size_t N1 = factors[0], N2 = N / N1;

    // complex_t *X;
    // X = new complex_t [N1*N2];
    float* Xr = new float[N1 * N2];
    float* Xi = new float[N1 * N2];

    /*for (size_t n1 = 0; n1 < N1; n1++)
        for (size_t n2 = 0; n2 < N2; n2++)
            X[n1*N2 + n2] = data[N1 * n2 + n1];
    */

    for (int n1 = 0; n1 < N1; n1++) {
        for (int n2 = 0; n2 < N2; n2++) {
            Xr[n1 * N2 + n2] = data[N1 * n2 + n1];
            Xi[n1 * N2 + n2] = data[N1 * n2 + n1 + N];
        }
    }

    for (size_t n1 = 0; n1 < N1; n1++) {
        std::vector<float> row;
        row.resize(N2 * 2);
        for (size_t i = 0; i < N2; i++) {
            // row.emplace_back(X[n1 * N2 + i]);
            row[i] = Xr[n1 * N2 + i];
            row[i + N2] = Xi[n1 * N2 + i];
        }
        std::vector<float> tmp;

        if (nextPowerOf2(row.size()) != row.size())
            tmp = hybridIFFT(row);
        else {
            tmp = StockhamIFFT(row);
            for (size_t i = 0; i < 2 * N2; i++) {
                tmp[i] /= N2;
            }
        }

        for (size_t n2 = 0; n2 < N2; n2++) {
            float tmpr = tmp[n2];
            float tmpi = tmp[n2 + N2];
            float wr = cosf(DOUBLE_PI * n1 * n2 / N);
            float wi = sinf(DOUBLE_PI * n1 * n2 / N);

            Xr[n1 * N2 + n2] = (tmpr * wr - tmpi * wi) * N2;
            Xi[n1 * N2 + n2] = (tmpr * wi + tmpi * wr) * N2;
        }
    }

    for (size_t n2 = 0; n2 < N2; n2++) {
        std::vector<float> col;
        col.resize(N1 * 2);
        for (size_t n1 = 0; n1 < N1; n1++)
        {
            // col.emplace_back(X[n1 * N2 + n2]);
            col[n1] = Xr[n1 * N2 + n2];
            col[n1 + N1] = Xi[n1 * N2 + n2];
        }

        std::vector<float> tmp;

        if (nextPowerOf2(col.size()) != col.size())
            tmp = hybridIFFT(col);
        else {
            tmp = StockhamIFFT(col);
            for (size_t i = 0; i < 2 * N1; i++) {
                tmp[i] /= N1;
            }
        }

        for (size_t n1 = 0; n1 < N1; n1++) {
            // X[n1 * N2 + n2] = tmp[n1];
            Xr[n1 * N2 + n2] = tmp[n1] * N1;
            Xi[n1 * N2 + n2] = tmp[n1 + N1] * N1;
        }
    }

    std::vector<float> result(data.size(), 0.0);
    for (size_t n1 = 0; n1 < N1; n1++)
        for (size_t n2 = 0; n2 < N2; n2++) {
            // result[N2 * n1 + n2] = X[n1 * N2 + n2];
            result[N2 * n1 + n2] = Xr[n1 * N2 + n2] / N;
            result[N2 * n1 + n2 + N] = Xi[n1 * N2 + n2] / N;
        }

    delete[] Xr;
    delete[] Xi;

    return result;
}
