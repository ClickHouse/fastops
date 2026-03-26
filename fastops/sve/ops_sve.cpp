#include "ops_sve.h"

#include <arm_sve.h>
#include <fastops/core/FastIntrinsics.h>

// getauxval(AT_HWCAP) is Linux-specific. SVE is currently only exposed on
// Linux AArch64 (not macOS, FreeBSD, or Windows ARM64), so we guard the
// runtime detection and return false on other platforms.
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif

namespace NFastOps {

bool HaveSve() {
#if defined(__linux__)
    static bool result = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    return result;
#else
    return false;
#endif
}

bool SveVectorLengthGt128() {
    static bool result = HaveSve() && svcntb() > 16;
    return result;
}

// SVE kernel wrappers — call the existing generic kernels with N=0 (SVE VLA tag).
// The macros EXP_PARAMS_COMMON, EXP_PARAMS_DOUBLE, etc. are defined in FastIntrinsics.h.

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveExpKernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
#define LOC_FUNC_NAME S_Pow<I_ElementSize, I_Exact>().template Calc<0, false>
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, LOC_FUNC_NAME, x, EXP_PARAMS_COMMON(c::c_1_over_ln_2), EXP_PARAMS_DOUBLE, EXP_PARAMS_FLOAT, EXP_PARAMS_APPROX);
#undef LOC_FUNC_NAME
}

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveExp2Kernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
#define LOC_FUNC_NAME S_Pow<I_ElementSize, I_Exact>().template Calc<0, false>
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, LOC_FUNC_NAME, x, EXP_PARAMS_COMMON(c::c_1_f), EXP_PARAMS_DOUBLE, EXP_PARAMS_FLOAT, EXP_PARAMS_APPROX);
#undef LOC_FUNC_NAME
}

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveExp10Kernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
#define LOC_FUNC_NAME S_Pow<I_ElementSize, I_Exact>().template Calc<0, false>
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, LOC_FUNC_NAME, x, EXP_PARAMS_COMMON(c::c_log2_10), EXP_PARAMS_DOUBLE, EXP_PARAMS_FLOAT, EXP_PARAMS_APPROX);
#undef LOC_FUNC_NAME
}

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveSigmoidKernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
#define LOC_FUNC_NAME S_Sigmoid<I_ElementSize, I_Exact>().template Calc<0, false>
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, LOC_FUNC_NAME, x, EXP_PARAMS_COMMON(c::c_neg_1_over_ln_2), EXP_PARAMS_DOUBLE, EXP_PARAMS_FLOAT, EXP_PARAMS_APPROX);
#undef LOC_FUNC_NAME
}

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveTanhKernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
#define LOC_FUNC_NAME S_Tanh<I_ElementSize, I_Exact>().template Calc<0, false>
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, LOC_FUNC_NAME, x, EXP_PARAMS_COMMON(c::c_neg_2_over_ln_2, c::c_2_f), EXP_PARAMS_DOUBLE, EXP_PARAMS_FLOAT, EXP_PARAMS_APPROX);
#undef LOC_FUNC_NAME
}

template <size_t I_ElementSize, bool I_Exact>
static FORCE_INLINE typename S_SIMDV<0, I_ElementSize>::t_f SveLnKernel(typename S_SIMDV<0, I_ElementSize>::t_f x) {
    AVX_FLOAT_MATH_FUNC_CALL_INTR(I_ElementSize, (LnV<0, I_ElementSize, I_Exact>), x, LN_PARAMS_COMMON, LN_PARAMS_DOUBLE, LN_PARAMS_FLOAT, LN_PARAMS_APPROX);
}

// VLA loop for float32 operations
template <bool I_Exact, bool I_OutAligned, class Kernel>
static void SveLoopF32(const float* from, size_t size, float* to, Kernel kernel) {
    for (size_t i = 0; i < size; i += svcntw()) {
        svbool_t pred = svwhilelt_b32(i, size);
        svfloat32_t x = svld1_f32(pred, from + i);
        svfloat32_t y = kernel(x);
        svst1_f32(pred, to + i, y);
    }
}

// VLA loop for float64 operations
template <bool I_Exact, bool I_OutAligned, class Kernel>
static void SveLoopF64(const double* from, size_t size, double* to, Kernel kernel) {
    for (size_t i = 0; i < size; i += svcntd()) {
        svbool_t pred = svwhilelt_b64(i, size);
        svfloat64_t x = svld1_f64(pred, from + i);
        svfloat64_t y = kernel(x);
        svst1_f64(pred, to + i, y);
    }
}

// Exp
template <bool I_Exact, bool I_OutAligned>
void ExpSve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveExpKernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void ExpSve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveExpKernel<8, I_Exact>(x); });
}

// Log
template <bool I_Exact, bool I_OutAligned>
void LogSve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveLnKernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void LogSve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveLnKernel<8, I_Exact>(x); });
}

// Sigmoid
template <bool I_Exact, bool I_OutAligned>
void SigmoidSve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveSigmoidKernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void SigmoidSve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveSigmoidKernel<8, I_Exact>(x); });
}

// Tanh
template <bool I_Exact, bool I_OutAligned>
void TanhSve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveTanhKernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void TanhSve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveTanhKernel<8, I_Exact>(x); });
}

// Exp2
template <bool I_Exact, bool I_OutAligned>
void Exp2Sve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveExp2Kernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void Exp2Sve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveExp2Kernel<8, I_Exact>(x); });
}

// Exp10
template <bool I_Exact, bool I_OutAligned>
void Exp10Sve(const float* from, size_t size, float* to) {
    SveLoopF32<I_Exact, I_OutAligned>(from, size, to, [](svfloat32_t x) { return SveExp10Kernel<4, I_Exact>(x); });
}
template <bool I_Exact, bool I_OutAligned>
void Exp10Sve(const double* from, size_t size, double* to) {
    SveLoopF64<I_Exact, I_OutAligned>(from, size, to, [](svfloat64_t x) { return SveExp10Kernel<8, I_Exact>(x); });
}

// Explicit instantiations
template void ExpSve<false, false>(const float*, size_t, float*);
template void ExpSve<false, true>(const float*, size_t, float*);
template void ExpSve<true, false>(const float*, size_t, float*);
template void ExpSve<true, true>(const float*, size_t, float*);

template void ExpSve<false, false>(const double*, size_t, double*);
template void ExpSve<false, true>(const double*, size_t, double*);
template void ExpSve<true, false>(const double*, size_t, double*);
template void ExpSve<true, true>(const double*, size_t, double*);

template void LogSve<false, false>(const float*, size_t, float*);
template void LogSve<false, true>(const float*, size_t, float*);
template void LogSve<true, false>(const float*, size_t, float*);
template void LogSve<true, true>(const float*, size_t, float*);

template void LogSve<false, false>(const double*, size_t, double*);
template void LogSve<false, true>(const double*, size_t, double*);
template void LogSve<true, false>(const double*, size_t, double*);
template void LogSve<true, true>(const double*, size_t, double*);

template void SigmoidSve<false, false>(const float*, size_t, float*);
template void SigmoidSve<false, true>(const float*, size_t, float*);
template void SigmoidSve<true, false>(const float*, size_t, float*);
template void SigmoidSve<true, true>(const float*, size_t, float*);

template void SigmoidSve<false, false>(const double*, size_t, double*);
template void SigmoidSve<false, true>(const double*, size_t, double*);
template void SigmoidSve<true, false>(const double*, size_t, double*);
template void SigmoidSve<true, true>(const double*, size_t, double*);

template void TanhSve<false, false>(const float*, size_t, float*);
template void TanhSve<false, true>(const float*, size_t, float*);
template void TanhSve<true, false>(const float*, size_t, float*);
template void TanhSve<true, true>(const float*, size_t, float*);

template void TanhSve<false, false>(const double*, size_t, double*);
template void TanhSve<false, true>(const double*, size_t, double*);
template void TanhSve<true, false>(const double*, size_t, double*);
template void TanhSve<true, true>(const double*, size_t, double*);

template void Exp2Sve<false, false>(const float*, size_t, float*);
template void Exp2Sve<false, true>(const float*, size_t, float*);
template void Exp2Sve<true, false>(const float*, size_t, float*);
template void Exp2Sve<true, true>(const float*, size_t, float*);

template void Exp2Sve<false, false>(const double*, size_t, double*);
template void Exp2Sve<false, true>(const double*, size_t, double*);
template void Exp2Sve<true, false>(const double*, size_t, double*);
template void Exp2Sve<true, true>(const double*, size_t, double*);

template void Exp10Sve<false, false>(const float*, size_t, float*);
template void Exp10Sve<false, true>(const float*, size_t, float*);
template void Exp10Sve<true, false>(const float*, size_t, float*);
template void Exp10Sve<true, true>(const float*, size_t, float*);

template void Exp10Sve<false, false>(const double*, size_t, double*);
template void Exp10Sve<false, true>(const double*, size_t, double*);
template void Exp10Sve<true, false>(const double*, size_t, double*);
template void Exp10Sve<true, true>(const double*, size_t, double*);

} // namespace NFastOps
