#pragma once

#include <cstddef>

namespace NFastOps {
    // Returns true if SVE is available on this CPU.
    bool HaveSve();

    // Returns true if the SVE vector length is strictly greater than 128 bits.
    bool SveVectorLengthGt128();

    template <bool I_Exact = false, bool I_OutAligned = false>
    void ExpSve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void ExpSve(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void LogSve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void LogSve(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void SigmoidSve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void SigmoidSve(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void TanhSve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void TanhSve(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp2Sve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp2Sve(const double* from, size_t size, double* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp10Sve(const float* from, size_t size, float* to);

    template <bool I_Exact = false, bool I_OutAligned = false>
    void Exp10Sve(const double* from, size_t size, double* to);
}
