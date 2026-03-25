#pragma once

#include <cstddef>

namespace NFastOps {
    template <bool I_Exact = false, bool I_OutAligned = false>
    void ExpNeon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void ExpNeon(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void LogNeon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void LogNeon(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void SigmoidNeon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void SigmoidNeon(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void TanhNeon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void TanhNeon(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp2Neon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp2Neon(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp10Neon(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp10Neon(const double* from, size_t size, double* to);
}
