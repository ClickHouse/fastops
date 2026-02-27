#include "ops_neon.h"

#include <fastops/core/FastIntrinsics.h>

namespace NFastOps {
    template <bool I_Exact, bool I_OutAligned>
    void ExpNeon(const float* from, size_t size, float* to) {
        NFastOps::AVXExp<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void ExpNeon(const double* from, size_t size, double* to) {
        NFastOps::AVXExp<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void LogNeon(const float* from, size_t size, float* to) {
        NFastOps::AVXLn<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void LogNeon(const double* from, size_t size, double* to) {
        NFastOps::AVXLn<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void SigmoidNeon(const float* from, size_t size, float* to) {
        NFastOps::AVXSigmoid<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void SigmoidNeon(const double* from, size_t size, double* to) {
        NFastOps::AVXSigmoid<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void TanhNeon(const float* from, size_t size, float* to) {
        NFastOps::AVXTanh<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void TanhNeon(const double* from, size_t size, double* to) {
        NFastOps::AVXTanh<I_Exact, I_OutAligned>(from, size, to);
    }

    template void ExpNeon<false, false>(const float* from, size_t size, float* to);
    template void ExpNeon<false, true>(const float* from, size_t size, float* to);
    template void ExpNeon<true, false>(const float* from, size_t size, float* to);
    template void ExpNeon<true, true>(const float* from, size_t size, float* to);

    template void ExpNeon<false, false>(const double* from, size_t size, double* to);
    template void ExpNeon<false, true>(const double* from, size_t size, double* to);
    template void ExpNeon<true, false>(const double* from, size_t size, double* to);
    template void ExpNeon<true, true>(const double* from, size_t size, double* to);

    template void LogNeon<false, false>(const float* from, size_t size, float* to);
    template void LogNeon<false, true>(const float* from, size_t size, float* to);
    template void LogNeon<true, false>(const float* from, size_t size, float* to);
    template void LogNeon<true, true>(const float* from, size_t size, float* to);

    template void LogNeon<false, false>(const double* from, size_t size, double* to);
    template void LogNeon<false, true>(const double* from, size_t size, double* to);
    template void LogNeon<true, false>(const double* from, size_t size, double* to);
    template void LogNeon<true, true>(const double* from, size_t size, double* to);

    template void SigmoidNeon<false, false>(const float* from, size_t size, float* to);
    template void SigmoidNeon<false, true>(const float* from, size_t size, float* to);
    template void SigmoidNeon<true, false>(const float* from, size_t size, float* to);
    template void SigmoidNeon<true, true>(const float* from, size_t size, float* to);

    template void SigmoidNeon<false, false>(const double* from, size_t size, double* to);
    template void SigmoidNeon<false, true>(const double* from, size_t size, double* to);
    template void SigmoidNeon<true, false>(const double* from, size_t size, double* to);
    template void SigmoidNeon<true, true>(const double* from, size_t size, double* to);

    template void TanhNeon<false, false>(const float* from, size_t size, float* to);
    template void TanhNeon<false, true>(const float* from, size_t size, float* to);
    template void TanhNeon<true, false>(const float* from, size_t size, float* to);
    template void TanhNeon<true, true>(const float* from, size_t size, float* to);

    template void TanhNeon<false, false>(const double* from, size_t size, double* to);
    template void TanhNeon<false, true>(const double* from, size_t size, double* to);
    template void TanhNeon<true, false>(const double* from, size_t size, double* to);
    template void TanhNeon<true, true>(const double* from, size_t size, double* to);

    template <bool I_Exact, bool I_OutAligned>
    void Exp2Neon(const float* from, size_t size, float* to) {
        NFastOps::AVXExp2<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void Exp2Neon(const double* from, size_t size, double* to) {
        NFastOps::AVXExp2<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void Exp10Neon(const float* from, size_t size, float* to) {
        NFastOps::AVXExp10<I_Exact, I_OutAligned>(from, size, to);
    }

    template <bool I_Exact, bool I_OutAligned>
    void Exp10Neon(const double* from, size_t size, double* to) {
        NFastOps::AVXExp10<I_Exact, I_OutAligned>(from, size, to);
    }

    template void Exp2Neon<false, false>(const float* from, size_t size, float* to);
    template void Exp2Neon<false, true>(const float* from, size_t size, float* to);
    template void Exp2Neon<true, false>(const float* from, size_t size, float* to);
    template void Exp2Neon<true, true>(const float* from, size_t size, float* to);

    template void Exp2Neon<false, false>(const double* from, size_t size, double* to);
    template void Exp2Neon<false, true>(const double* from, size_t size, double* to);
    template void Exp2Neon<true, false>(const double* from, size_t size, double* to);
    template void Exp2Neon<true, true>(const double* from, size_t size, double* to);

    template void Exp10Neon<false, false>(const float* from, size_t size, float* to);
    template void Exp10Neon<false, true>(const float* from, size_t size, float* to);
    template void Exp10Neon<true, false>(const float* from, size_t size, float* to);
    template void Exp10Neon<true, true>(const float* from, size_t size, float* to);

    template void Exp10Neon<false, false>(const double* from, size_t size, double* to);
    template void Exp10Neon<false, true>(const double* from, size_t size, double* to);
    template void Exp10Neon<true, false>(const double* from, size_t size, double* to);
    template void Exp10Neon<true, true>(const double* from, size_t size, double* to);
}
