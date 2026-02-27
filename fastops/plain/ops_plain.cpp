#include "ops_plain.h"

#include <contrib/libs/fmath/fmath.hpp>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) || defined(__i386__) || defined(_M_IX86)
#include <xmmintrin.h>
#endif

namespace NFastOps {
    void ExpPlain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            // double, because float returns nan instead of inf on higher values
            // (and float works slower than double on random data!)
            to[i] = fmath::expd(from[i]);
        }
    }

    void ExpPlain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = fmath::expd(from[i]);
        }
    }

    void LogPlain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            // there is no exact fast float log
            to[i] = log(from[i]);
        }
    }

    void LogPlain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            // there is no exact fast double log
            to[i] = log(from[i]);
        }
    }

    void SigmoidPlain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = 1.0 / (1.0f + (float)fmath::expd(-from[i]));
        }
    }

    void SigmoidPlain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = 1.0 / (1.0 + fmath::expd(-from[i]));
        }
    }

    void TanhPlain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = 2.0f / (1.0f + (float)fmath::expd(-2 * from[i])) - 1.0f;
        }
    }

    void TanhPlain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = 2 / (1.0 + fmath::expd(-2 * from[i])) - 1;
        }
    }

    void Exp2Plain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = fmath::expd(from[i] * 0.6931471805599453094172321214581765680755);
        }
    }

    void Exp2Plain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = fmath::expd(from[i] * 0.6931471805599453094172321214581765680755);
        }
    }

    void Exp10Plain(const float* from, size_t size, float* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = fmath::expd(from[i] * 2.302585092994045684017991454684364207601);
        }
    }

    void Exp10Plain(const double* from, size_t size, double* to) {
        for (size_t i = 0; i < size; ++i) {
            to[i] = fmath::expd(from[i] * 2.302585092994045684017991454684364207601);
        }
    }
}
