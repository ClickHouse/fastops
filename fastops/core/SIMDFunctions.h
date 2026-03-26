#pragma once

#ifndef SIMD_FUNCTIONS_H
#define SIMD_FUNCTIONS_H

#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64) || defined(__i386__) || defined(_M_IX86)
#define FASTOPS_X86
#elif defined(__aarch64__)
#define FASTOPS_ARM64
#endif

#ifdef FASTOPS_X86

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <emmintrin.h>
#include <xmmintrin.h>
#include <immintrin.h>

#if defined(__clang_major__) && !defined(__apple_build_version__) && __clang_major__ >= 8 || __GNUC__  >= 11
#   define Y_HAVE_NEW_INTRINSICS
#endif

#if !defined(Y_HAVE_NEW_INTRINSICS) && __GNUC__ < 11
static __inline__ __m128i
_mm_loadu_si32(void const* __a) {
    struct __loadu_si32 {
        int __v;
    } __attribute__((__packed__, __may_alias__));
    int __u = ((struct __loadu_si32*)__a)->__v;
    return __extension__(__m128i)(__v4si){__u, 0, 0, 0};
}
#endif

#if !defined(__clang__) && __GNUC__ < 9
static __inline__ __m128i
_mm_loadu_si64(void const* __a) {
    struct __loadu_si64 {
        long long __v;
    } __attribute__((__packed__, __may_alias__));
    long long __u = ((struct __loadu_si64*)__a)->__v;
    return (__m128i){__u, 0L};
}
#endif

#if !defined(Y_HAVE_NEW_INTRINSICS)
static __inline__ void
_mm_storeu_si32(void const* __p, __m128i __b) {
    struct __storeu_si32 {
        int __v;
    } __attribute__((__packed__, __may_alias__));
    ((struct __storeu_si32*)__p)->__v = ((__v4si)__b)[0];
}

static __inline__ void
_mm_storeu_si64(void const* __p, __m128i __b) {
    struct __storeu_si64 {
        long long __v;
    } __attribute__((__packed__, __may_alias__));
    ((struct __storeu_si64*)__p)->__v = ((__v2di)__b)[0];
}

static __inline__ void
_mm_storeu_si16(void const* __p, __m128i __b) {
    struct __storeu_si16 {
        short __v;
    } __attribute__((__packed__, __may_alias__));
    ((struct __storeu_si16*)__p)->__v = ((__v8hi)__b)[0];
}
#endif
#endif

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#else
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#ifdef _MSC_VER
#include <stdint.h>
#endif

namespace NFastOps {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Function returns a ymm register with all floats set to 1.f
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FORCE_INLINE __m256 YMMOneFloat() noexcept {
        return _mm256_set1_ps(1.f);
    }
    FORCE_INLINE __m256i YMMOneFloatSI() noexcept {
        return _mm256_castps_si256(YMMOneFloat());
    }
    FORCE_INLINE __m256d YMMOneDouble() noexcept {
        return _mm256_set1_pd(1.);
    }
    FORCE_INLINE __m256i YMMOneDoubleSI() noexcept {
        return _mm256_castpd_si256(YMMOneDouble());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Function calculates horizontal sum of the YMM register. Assumes floats are stored.
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FORCE_INLINE __m128 SumYMM(__m128 x) noexcept {
        const __m128 hi_dual = _mm_movehl_ps(x, x);
        const __m128 sum_dual = _mm_add_ps(x, hi_dual);
        const __m128 hi = _mm_shuffle_ps(sum_dual, sum_dual, 0x1);
        return _mm_add_ss(sum_dual, hi);
    }
    FORCE_INLINE __m128 SumYMM(__m256 x) noexcept {
        const __m128 hi_quad = _mm256_extractf128_ps(x, 1); // hi_quad = ( x7, x6, x5, x4 )
        return SumYMM(_mm_add_ps(_mm256_castps256_ps128(x), hi_quad));
    }
    FORCE_INLINE float SumYMMR(__m128 x) noexcept {
        return _mm_cvtss_f32(SumYMM(x));
    }
    FORCE_INLINE float SumYMMR(__m256 x) noexcept {
        return _mm_cvtss_f32(SumYMM(x));
    }

    FORCE_INLINE __m128d SumYMM(__m128d x) noexcept {
        const __m128d hi_dual = _mm_castps_pd(_mm_movehl_ps(_mm_castpd_ps(x), _mm_castpd_ps(x)));
        return _mm_add_sd(x, hi_dual);
    }
    FORCE_INLINE __m128d SumYMM(__m256d x) noexcept {
        const __m128d hi_quad = _mm256_extractf128_pd(x, 1);
        return SumYMM(_mm_add_pd(_mm256_castpd256_pd128(x), hi_quad));
    }
    FORCE_INLINE double SumYMMR(__m128d x) noexcept {
        return _mm_cvtsd_f64(SumYMM(x));
    }
    FORCE_INLINE double SumYMMR(__m256d x) noexcept {
        return _mm_cvtsd_f64(SumYMM(x));
    }

//#######################################################################################################################################################################
// Macro - no other good way :-(
#define OPERATE_SEPARATELY_I(op, v, param) \
    _mm256_permute2f128_si256(_mm256_castsi128_si256(op(_mm256_castsi256_si128(v), param)), _mm256_castsi128_si256(op(_mm256_extractf128_si256(v, 1), param)), 32);

#define OPERATE_ELEMENTWISE_I(op, v1, v2)                                                             \
    _mm256_permute2f128_si256(                                                                        \
        _mm256_castsi128_si256(op(_mm256_castsi256_si128(v1), _mm256_castsi256_si128(v2))),           \
        _mm256_castsi128_si256(op(_mm256_extractf128_si256(v1, 1), _mm256_extractf128_si256(v2, 1))), \
        32);

#define FMADD_NO_AVX2()                                      \
    FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {  \
        return Add(Mul(v1, v2), v3);                         \
    }                                                        \
    FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {  \
        return Sub(Mul(v1, v2), v3);                         \
    }                                                        \
    FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { \
        return Sub(v3, Mul(v1, v2));                         \
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //  These functions only work for inputs in the range: [-2^51, 2^51]
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    FORCE_INLINE __m128i double_to_int64(__m128d x) {
        const __m256d magic_cvt_c = _mm256_set1_pd(0x0018000000000000);
        x = _mm_add_pd(x, _mm256_castpd256_pd128(magic_cvt_c));
        return _mm_sub_epi64(_mm_castpd_si128(x), _mm_castpd_si128(_mm256_castpd256_pd128(magic_cvt_c)));
    }
    FORCE_INLINE __m256i double_to_int64(__m256d x) {
        const __m256d magic_cvt_c = _mm256_set1_pd(0x0018000000000000);
        x = _mm256_add_pd(x, magic_cvt_c);
#ifndef NO_AVX2
        return _mm256_sub_epi64(_mm256_castpd_si256(x), _mm256_castpd_si256(magic_cvt_c));
#else
        return OPERATE_SEPARATELY_I(_mm_sub_epi64, _mm256_castpd_si256(x), _mm256_castsi256_si128(_mm256_castpd_si256(magic_cvt_c)));
#endif
    }
    FORCE_INLINE __m128d i64o_double(__m128i x) {
        const __m256d magic_cvt_c = _mm256_set1_pd(0x0018000000000000);
        x = _mm_add_epi64(x, _mm_castpd_si128(_mm256_castpd256_pd128(magic_cvt_c)));
        return _mm_sub_pd(_mm_castsi128_pd(x), _mm256_castpd256_pd128(magic_cvt_c));
    }
    FORCE_INLINE __m256d i64o_double(__m256i x) {
        const __m256d magic_cvt_c = _mm256_set1_pd(0x0018000000000000);
#ifndef NO_AVX2
        x = _mm256_add_epi64(x, _mm256_castpd_si256(magic_cvt_c));
#else
        x = OPERATE_SEPARATELY_I(_mm_add_epi64, x, _mm256_castsi256_si128(_mm256_castpd_si256(magic_cvt_c)));
#endif
        return _mm256_sub_pd(_mm256_castsi256_pd(x), magic_cvt_c);
    }

    //#######################################################################################################################################################################

    template <size_t I_NOfElements, size_t I_ElemSize>
    struct S_SIMDV;

    struct S_SIMDSmallBaseF {
        using t_i = __m128i;
        using t_f = __m128;
        using t_base_type = float;
        //using t_type_d = __m128d;

        FORCE_INLINE static t_i Cast(__m256i v) noexcept {
            return _mm256_castsi256_si128(v);
        }
        FORCE_INLINE static t_f Cast(__m256 v) noexcept {
            return _mm256_castps256_ps128(v);
        }
        //FORCE_INLINE static t_type_d Cast(__m256d v) noexcept { return _mm256_castpd256_pd128(v); }
        FORCE_INLINE static t_i Cast(t_i v) noexcept {
            return v;
        }
        FORCE_INLINE static t_f Cast(t_f v) noexcept {
            return v;
        }
        //FORCE_INLINE static t_type_d Cast(t_type_d v) noexcept { return v; }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_i SetZeroI() {
            return _mm_setzero_si128();
        }
        FORCE_INLINE static t_f SetZeroF() {
            return _mm_setzero_ps();
        }
        FORCE_INLINE static t_f Set1(float v) {
            return _mm_set1_ps(v);
        }
        FORCE_INLINE static t_f Set(float v1, float v2, float v3, float v4) {
            return _mm_set_ps(v1, v2, v3, v4);
        }
        FORCE_INLINE static t_i Set1(int v) {
            return _mm_set1_epi32(v);
        }
        FORCE_INLINE static t_i Set(int v1, int v2, int v3, int v4) {
            return _mm_set_epi32(v1, v2, v3, v4);
        }

        FORCE_INLINE static t_i CastI(t_f v) {
            return _mm_castps_si128(v);
        }
        FORCE_INLINE static t_f CastF(t_i v) {
            return _mm_castsi128_ps(v);
        }
        FORCE_INLINE static t_f CVTI2F(t_i v) {
            return _mm_cvtepi32_ps(v);
        }
        FORCE_INLINE static t_i CVTF2I(t_f v) {
            return _mm_cvtps_epi32(v);
        }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return _mm_cmpeq_epi32(v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return _mm_srli_epi32(v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return _mm_slli_epi32(v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return _mm_srai_epi32(v, i);
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            return _mm_testc_ps(v1, v2);
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            return _mm_testz_ps(v1, v2);
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return _mm_and_ps(v1, v2);
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return _mm_andnot_ps(v1, v2);
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return _mm_or_ps(v1, v2);
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return _mm_xor_ps(v1, v2);
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return _mm_sub_epi32(v1, v2);
        }
        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            return _mm_blendv_ps(v1, v2, v3);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            return _mm_cmp_ps(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return _mm_add_epi32(v1, v2);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct S_SIMDSmallMultiple: public S_SIMDSmallBaseF {
        using S_SIMDSmallBaseF::Add;
        using S_SIMDSmallBaseF::Sub;
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm_cmp_ps(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm_cmpeq_ps(v1, v2);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm_mul_ps(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm_div_ps(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm_add_ps(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm_sub_ps(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm_sqrt_ps(v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fmadd_ps(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm_fmsub_ps(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fnmadd_ps(v1, v2, v3);
        }
#else
        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm_min_ps(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm_max_ps(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm_floor_ps(v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<1, 4>: public S_SIMDSmallBaseF {
        using S_SIMDSmallBaseF::Add;
        using S_SIMDSmallBaseF::Sub;

        FORCE_INLINE static t_f LoadU(const float* p) {
            return _mm_load_ss(p);
        }
        FORCE_INLINE static t_f Load(const float* p) {
            return _mm_load_ss(p);
        }
        FORCE_INLINE static t_i LoadU(const int* p) {
            return _mm_loadu_si32(p);
        }
        FORCE_INLINE static t_i Load(const int* p) {
            return _mm_loadu_si32(p);
        }
        FORCE_INLINE static void StoreU(float* p, t_f v) {
            return _mm_store_ss(p, v);
        }
        FORCE_INLINE static void Store(float* p, t_f v) {
            return _mm_store_ss(p, v);
        }
        FORCE_INLINE static void StoreU(int* p, t_i v) {
            return _mm_storeu_si32(p, v);
        }
        FORCE_INLINE static void Store(int* p, t_i v) {
            return _mm_storeu_si32(p, v);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm_cmp_ss(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm_cmpeq_ss(v1, v2);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm_mul_ss(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm_div_ss(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm_add_ss(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm_sub_ss(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm_sqrt_ss(v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fmadd_ss(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm_fmsub_ss(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fnmadd_ss(v1, v2, v3);
        }
#else
        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm_min_ss(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm_max_ss(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm_floor_ss(v, v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<2, 4>: public S_SIMDSmallMultiple {
        FORCE_INLINE static t_f LoadU(const float* p) {
            return _mm_castpd_ps(_mm_load_sd((const double*)p));
        }
        FORCE_INLINE static t_f Load(const float* p) {
            return _mm_castpd_ps(_mm_load_sd((const double*)p));
        }
        FORCE_INLINE static t_i LoadU(const int* p) {
            return _mm_loadu_si64(p);
        }
        FORCE_INLINE static t_i Load(const int* p) {
            return _mm_loadu_si64(p);
        }
        FORCE_INLINE static void StoreU(float* p, t_f v) {
            return _mm_store_sd((double*)p, _mm_castps_pd(v));
        }
        FORCE_INLINE static void Store(float* p, t_f v) {
            return _mm_store_sd((double*)p, _mm_castps_pd(v));
        }
        FORCE_INLINE static void StoreU(int* p, t_i v) {
            return _mm_storeu_si64(p, v);
        }
        FORCE_INLINE static void Store(int* p, t_i v) {
            return _mm_storeu_si64(p, v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<4, 4>: public S_SIMDSmallMultiple {
        FORCE_INLINE static t_f LoadU(const float* p) {
            return _mm_loadu_ps(p);
        }
        FORCE_INLINE static t_f Load(const float* p) {
            return _mm_load_ps(p);
        }
        FORCE_INLINE static t_i LoadU(const int* p) {
            return _mm_loadu_si128((const t_i*)p);
        }
        FORCE_INLINE static t_i Load(const int* p) {
            return _mm_load_si128((const t_i*)p);
        }
        FORCE_INLINE static void StoreU(float* p, t_f v) {
            return _mm_storeu_ps(p, v);
        }
        FORCE_INLINE static void Store(float* p, t_f v) {
            return _mm_store_ps(p, v);
        }
        FORCE_INLINE static void StoreU(int* p, t_i v) {
            return _mm_storeu_si128((t_i*)p, v);
        }
        FORCE_INLINE static void Store(int* p, t_i v) {
            return _mm_store_si128((t_i*)p, v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<8, 4> {
        using t_i = __m256i;
        using t_f = __m256;
        using t_base_type = float;
        //using t_type_d = __m256d;

        FORCE_INLINE static t_i Cast(t_i v) noexcept {
            return v;
        }
        FORCE_INLINE static t_f Cast(t_f v) noexcept {
            return v;
        }
        //FORCE_INLINE static t_type_d Cast(t_type_d v) noexcept { return v; }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_f LoadU(const float* p) {
            return _mm256_loadu_ps(p);
        }
        FORCE_INLINE static t_f Load(const float* p) {
            return _mm256_load_ps(p);
        }
        FORCE_INLINE static t_i LoadU(const int* p) {
            return _mm256_loadu_si256((const t_i*)p);
        }
        FORCE_INLINE static t_i Load(const int* p) {
            return _mm256_load_si256((const t_i*)p);
        }
        FORCE_INLINE static void StoreU(float* p, t_f v) {
            return _mm256_storeu_ps(p, v);
        }
        FORCE_INLINE static void Store(float* p, t_f v) {
            return _mm256_store_ps(p, v);
        }
        FORCE_INLINE static void StoreU(int* p, t_i v) {
            return _mm256_storeu_si256((t_i*)p, v);
        }
        FORCE_INLINE static void Store(int* p, t_i v) {
            return _mm256_store_si256((t_i*)p, v);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_i SetZeroI() {
            return _mm256_setzero_si256();
        }
        FORCE_INLINE static t_f SetZeroF() {
            return _mm256_setzero_ps();
        }
        FORCE_INLINE static t_f Set1(float v) {
            return _mm256_set1_ps(v);
        }
        FORCE_INLINE static t_i Set1(int v) {
            return _mm256_set1_epi32(v);
        }
        FORCE_INLINE static t_f Set(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8) {
            return _mm256_set_ps(v1, v2, v3, v4, v5, v6, v7, v8);
        }
        FORCE_INLINE static t_i Set(int v1, int v2, int v3, int v4, int v5, int v6, int v7, int v8) {
            return _mm256_set_epi32(v1, v2, v3, v4, v5, v6, v7, v8);
        }

        FORCE_INLINE static t_i CastI(t_f v) {
            return _mm256_castps_si256(v);
        }
        FORCE_INLINE static t_f CastF(t_i v) {
            return _mm256_castsi256_ps(v);
        }
        FORCE_INLINE static t_f CVTI2F(t_i v) {
            return _mm256_cvtepi32_ps(v);
        }
        FORCE_INLINE static t_i CVTF2I(t_f v) {
            return _mm256_cvtps_epi32(v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return _mm256_cmpeq_epi32(v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return _mm256_srli_epi32(v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return _mm256_slli_epi32(v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return _mm256_srai_epi32(v, i);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return _mm256_add_epi32(v1, v2);
        }
        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return _mm256_sub_epi32(v1, v2);
        }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm256_fmadd_ps(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm256_fmsub_ps(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm256_fnmadd_ps(v1, v2, v3);
        }
#else
        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_cmpeq_epi32, v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_srli_epi32, v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_slli_epi32, v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_srai_epi32, v, i);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_add_epi32, v1, v2);
        }
        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_sub_epi32, v1, v2);
        }

        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            return _mm256_testc_ps(v1, v2);
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            return _mm256_testz_ps(v1, v2);
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return _mm256_and_ps(v1, v2);
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return _mm256_andnot_ps(v1, v2);
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return _mm256_or_ps(v1, v2);
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return _mm256_xor_ps(v1, v2);
        }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            return _mm256_blendv_ps(v1, v2, v3);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            return _mm256_cmp_ps(v1, v2, I_Mode);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm256_cmp_ps(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm256_cmp_ps(v1, v2, _CMP_EQ_OQ);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm256_mul_ps(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm256_div_ps(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm256_add_ps(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm256_sub_ps(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm256_sqrt_ps(v);
        }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm256_min_ps(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm256_max_ps(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm256_floor_ps(v);
        }
    };

    //############################################################################################################################################################

    struct S_SIMDSmallBaseD {
        using t_i = __m128i;
        using t_f = __m128d;
        using t_base_type = double;
        //using t_type_d = __m128d;

        FORCE_INLINE static t_i Cast(__m256i v) noexcept {
            return _mm256_castsi256_si128(v);
        }
        FORCE_INLINE static t_f Cast(__m256d v) noexcept {
            return _mm256_castpd256_pd128(v);
        }
        FORCE_INLINE static t_i Cast(t_i v) noexcept {
            return v;
        }
        FORCE_INLINE static t_f Cast(t_f v) noexcept {
            return v;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_i SetZeroI() {
            return _mm_setzero_si128();
        }
        FORCE_INLINE static t_f SetZeroF() {
            return _mm_setzero_pd();
        }
        FORCE_INLINE static t_f Set1(double v) {
            return _mm_set1_pd(v);
        }
        FORCE_INLINE static t_f Set(double v1, double v2) {
            return _mm_set_pd(v1, v2);
        }
        FORCE_INLINE static t_i Set1(int64_t v) {
            return _mm_set1_epi64x(v);
        }
        FORCE_INLINE static t_i Set(int64_t v1, int64_t v2) {
            return _mm_set_epi64x(v1, v2);
        }

        FORCE_INLINE static t_i CastI(t_f v) {
            return _mm_castpd_si128(v);
        }
        FORCE_INLINE static t_f CastF(t_i v) {
            return _mm_castsi128_pd(v);
        }
        FORCE_INLINE static t_f CVTI2F(t_i v) {
            return /*_mm_cvtepi64_pd(v);*/ i64o_double(v);
        }
        FORCE_INLINE static t_i CVTF2I(t_f v) {
            return /*_mm_cvtpd_epi64(v);*/ double_to_int64(v);
        }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return _mm_cmpeq_epi64(v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return _mm_srli_epi64(v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return _mm_slli_epi64(v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return _mm_srai_epi32(v, i);
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            return _mm_testc_pd(v1, v2);
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            return _mm_testz_pd(v1, v2);
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return _mm_and_pd(v1, v2);
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return _mm_andnot_pd(v1, v2);
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return _mm_or_pd(v1, v2);
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return _mm_xor_pd(v1, v2);
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return _mm_sub_epi64(v1, v2);
        }
        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            return _mm_blendv_pd(v1, v2, v3);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            return _mm_cmp_pd(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return _mm_add_epi64(v1, v2);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct S_SIMDSmallMultipleD: public S_SIMDSmallBaseD {
        using S_SIMDSmallBaseD::Add;
        using S_SIMDSmallBaseD::Sub;
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm_cmp_pd(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm_cmpeq_pd(v1, v2);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm_mul_pd(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm_div_pd(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm_add_pd(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm_sub_pd(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm_sqrt_pd(v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fmadd_pd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm_fmsub_pd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fnmadd_pd(v1, v2, v3);
        }
#else
        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm_min_pd(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm_max_pd(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm_floor_pd(v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<1, 8>: public S_SIMDSmallBaseD {
        using S_SIMDSmallBaseD::Add;
        using S_SIMDSmallBaseD::Sub;

        FORCE_INLINE static t_f LoadU(const double* p) {
            return _mm_load_sd(p);
        }
        FORCE_INLINE static t_f Load(const double* p) {
            return _mm_load_sd(p);
        }
        FORCE_INLINE static t_i LoadU(const int64_t* p) {
            return _mm_loadu_si64(p);
        }
        FORCE_INLINE static t_i Load(const int64_t* p) {
            return _mm_loadu_si64(p);
        }
        FORCE_INLINE static void StoreU(double* p, t_f v) {
            return _mm_store_sd(p, v);
        }
        FORCE_INLINE static void Store(double* p, t_f v) {
            return _mm_store_sd(p, v);
        }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) {
            return _mm_storeu_si64(p, v);
        }
        FORCE_INLINE static void Store(int64_t* p, t_i v) {
            return _mm_storeu_si64(p, v);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm_cmp_sd(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm_cmpeq_sd(v1, v2);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm_mul_sd(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm_div_sd(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm_add_sd(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm_sub_sd(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm_sqrt_sd(v, v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fmadd_sd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm_fmsub_sd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm_fnmadd_sd(v1, v2, v3);
        }
#else
        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm_min_sd(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm_max_sd(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm_floor_sd(v, v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<2, 8>: public S_SIMDSmallMultipleD {
        FORCE_INLINE static t_f LoadU(const double* p) {
            return _mm_castps_pd(_mm_loadu_ps((const float*)p));
        }
        FORCE_INLINE static t_f Load(const double* p) {
            return _mm_castps_pd(_mm_load_ps((const float*)p));
        }
        FORCE_INLINE static t_i LoadU(const int64_t* p) {
            return _mm_loadu_si128((const t_i*)p);
        }
        FORCE_INLINE static t_i Load(const int64_t* p) {
            return _mm_load_si128((const t_i*)p);
        }
        FORCE_INLINE static void StoreU(double* p, t_f v) {
            return _mm_storeu_ps((float*)p, _mm_castpd_ps(v));
        }
        FORCE_INLINE static void Store(double* p, t_f v) {
            return _mm_store_ps((float*)p, _mm_castpd_ps(v));
        }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) {
            return _mm_storeu_si128((t_i*)p, v);
        }
        FORCE_INLINE static void Store(int64_t* p, t_i v) {
            return _mm_store_si128((t_i*)p, v);
        }
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template <>
    struct S_SIMDV<4, 8> {
        using t_i = __m256i;
        using t_f = __m256d;
        using t_base_type = double;

        FORCE_INLINE static t_i Cast(t_i v) noexcept {
            return v;
        }
        FORCE_INLINE static t_f Cast(t_f v) noexcept {
            return v;
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_f LoadU(const double* p) {
            return _mm256_loadu_pd(p);
        }
        FORCE_INLINE static t_f Load(const double* p) {
            return _mm256_load_pd(p);
        }
        FORCE_INLINE static t_i LoadU(const int64_t* p) {
            return _mm256_loadu_si256((const t_i*)p);
        }
        FORCE_INLINE static t_i Load(const int64_t* p) {
            return _mm256_load_si256((const t_i*)p);
        }
        FORCE_INLINE static void StoreU(double* p, t_f v) {
            return _mm256_storeu_pd(p, v);
        }
        FORCE_INLINE static void Store(double* p, t_f v) {
            return _mm256_store_pd(p, v);
        }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) {
            return _mm256_storeu_si256((t_i*)p, v);
        }
        FORCE_INLINE static void Store(int64_t* p, t_i v) {
            return _mm256_store_si256((t_i*)p, v);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        FORCE_INLINE static t_i SetZeroI() {
            return _mm256_setzero_si256();
        }
        FORCE_INLINE static t_f SetZeroF() {
            return _mm256_setzero_pd();
        }
        FORCE_INLINE static t_f Set1(double v) {
            return _mm256_set1_pd(v);
        }
        FORCE_INLINE static t_i Set1(int64_t v) {
            return _mm256_set1_epi64x(v);
        }
        FORCE_INLINE static t_f Set(double v1, double v2, double v3, double v4) {
            return _mm256_set_pd(v1, v2, v3, v4);
        }
        FORCE_INLINE static t_i Set(int64_t v1, int64_t v2, int64_t v3, int64_t v4) {
            return _mm256_set_epi64x(v1, v2, v3, v4);
        }

        FORCE_INLINE static t_i CastI(t_f v) {
            return _mm256_castpd_si256(v);
        }
        FORCE_INLINE static t_f CastF(t_i v) {
            return _mm256_castsi256_pd(v);
        }
        FORCE_INLINE static t_f CVTI2F(t_i v) {
            return /*_mm256_cvtepi64_pd(v);*/ i64o_double(v);
        }
        FORCE_INLINE static t_i CVTF2I(t_f v) {
            return /*_mm256_cvtpd_epi64(v);*/ double_to_int64(v);
        }

#ifndef NO_AVX2
        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return _mm256_cmpeq_epi64(v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return _mm256_srli_epi64(v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return _mm256_slli_epi64(v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return _mm256_srai_epi32(v, i);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return _mm256_add_epi64(v1, v2);
        }
        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return _mm256_sub_epi64(v1, v2);
        }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) {
            return _mm256_fmadd_pd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) {
            return _mm256_fmsub_pd(v1, v2, v3);
        }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) {
            return _mm256_fnmadd_pd(v1, v2, v3);
        }
#else
        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_cmpeq_epi64, v1, v2);
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_srli_epi64, v, i);
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_slli_epi64, v, i);
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return OPERATE_SEPARATELY_I(_mm_srai_epi32, v, i);
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_add_epi64, v1, v2);
        }
        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) {
            return OPERATE_ELEMENTWISE_I(_mm_sub_epi64, v1, v2);
        }

        FMADD_NO_AVX2();
#endif

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            return _mm256_testc_pd(v1, v2);
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            return _mm256_testz_pd(v1, v2);
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return _mm256_and_pd(v1, v2);
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return _mm256_andnot_pd(v1, v2);
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return _mm256_or_pd(v1, v2);
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return _mm256_xor_pd(v1, v2);
        }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            return _mm256_blendv_pd(v1, v2, v3);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            return _mm256_cmp_pd(v1, v2, I_Mode);
        }

        ///////////////////////////////////////////////////////////////////////////////////////////////////////

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            return _mm256_cmp_pd(v1, v2, I_Mode);
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return _mm256_cmp_pd(v1, v2, _CMP_EQ_OQ);
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) {
            return _mm256_mul_pd(v1, v2);
        }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) {
            return _mm256_div_pd(v1, v2);
        }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) {
            return _mm256_add_pd(v1, v2);
        }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) {
            return _mm256_sub_pd(v1, v2);
        }
        FORCE_INLINE static t_f Sqrt(t_f v) {
            return _mm256_sqrt_pd(v);
        }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) {
            return _mm256_min_pd(v1, v2);
        }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) {
            return _mm256_max_pd(v1, v2);
        }
        FORCE_INLINE static t_f Floor(t_f v) {
            return _mm256_floor_pd(v);
        }
    };

    template <size_t I_ElemSize>
    using S_MaxSIMD = S_SIMDV<32 / I_ElemSize, I_ElemSize>;
}

#elif defined(FASTOPS_ARM64)

#include <arm_neon.h>

#ifndef FORCE_INLINE
#define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#define _CMP_EQ_OQ 0
#define _CMP_GT_OQ 14

namespace NFastOps {

    FORCE_INLINE int64x2_t double_to_int64(float64x2_t x) {
        return vcvtnq_s64_f64(x);
    }
    FORCE_INLINE float64x2_t i64o_double(int64x2_t x) {
        return vcvtq_f64_s64(x);
    }

    template <size_t I_NOfElements, size_t I_ElemSize>
    struct S_SIMDV;

    struct S_NeonBaseF {
        using t_i = int32x4_t;
        using t_f = float32x4_t;
        using t_base_type = float;

        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { return vdupq_n_s32(0); }
        FORCE_INLINE static t_f SetZeroF() { return vdupq_n_f32(0); }
        FORCE_INLINE static t_f Set1(float v) { return vdupq_n_f32(v); }
        FORCE_INLINE static t_f Set(float v1, float v2, float v3, float v4) {
            float __attribute__((aligned(16))) data[4] = {v4, v3, v2, v1};
            return vld1q_f32(data);
        }
        FORCE_INLINE static t_i Set1(int v) { return vdupq_n_s32(v); }
        FORCE_INLINE static t_i Set(int v1, int v2, int v3, int v4) {
            int __attribute__((aligned(16))) data[4] = {v4, v3, v2, v1};
            return vld1q_s32(data);
        }

        FORCE_INLINE static t_i CastI(t_f v) { return vreinterpretq_s32_f32(v); }
        FORCE_INLINE static t_f CastF(t_i v) { return vreinterpretq_f32_s32(v); }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return vcvtq_f32_s32(v); }
        FORCE_INLINE static t_i CVTF2I(t_f v) { return vcvtnq_s32_f32(v); }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return vreinterpretq_s32_u32(vceqq_s32(v1, v2));
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v), vdupq_n_s32(-i)));
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return vshlq_s32(v, vdupq_n_s32(i));
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return vshlq_s32(v, vdupq_n_s32(-i));
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            uint32x4_t bits = vbicq_u32(vreinterpretq_u32_f32(v2), vreinterpretq_u32_f32(v1));
            return vmaxvq_u32(bits) == 0;
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            uint32x4_t bits = vandq_u32(vreinterpretq_u32_f32(v1), vreinterpretq_u32_f32(v2));
            return vmaxvq_u32(bits) == 0;
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1), vreinterpretq_u32_f32(v2)));
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v2), vreinterpretq_u32_f32(v1)));
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v1), vreinterpretq_u32_f32(v2)));
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1), vreinterpretq_u32_f32(v2)));
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return vsubq_s32(v1, v2); }
        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            // Broadcast sign bit to all 32 bits, matching x86 blendv semantics:
            // sign=1 (negative) selects from v2, sign=0 (non-negative) selects from v1.
            uint32x4_t mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(v3), 31));
            return vbslq_f32(mask, v2, v1);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f32_u32(vceqq_f32(v1, v2));
            else
                return vreinterpretq_f32_u32(vcgtq_f32(v1, v2));
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return vaddq_s32(v1, v2); }
    };

    struct S_NeonMultipleF : public S_NeonBaseF {
        using S_NeonBaseF::Add;
        using S_NeonBaseF::Sub;
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f32_u32(vceqq_f32(v1, v2));
            else
                return vreinterpretq_f32_u32(vcgtq_f32(v1, v2));
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(vceqq_f32(v1, v2));
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return vmulq_f32(v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return vdivq_f32(v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return vaddq_f32(v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return vsubq_f32(v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return vsqrtq_f32(v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return vfmaq_f32(v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return vfmaq_f32(vnegq_f32(v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return vfmsq_f32(v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return vminq_f32(v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return vmaxq_f32(v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return vrndmq_f32(v); }
    };

    // S_SIMDV<1,4>: load 1 float, zero upper lanes
    template <>
    struct S_SIMDV<1, 4> : public S_NeonBaseF {
        using S_NeonBaseF::Add;
        using S_NeonBaseF::Sub;

        FORCE_INLINE static t_f LoadU(const float* p) { return vsetq_lane_f32(*p, vdupq_n_f32(0), 0); }
        FORCE_INLINE static t_f Load(const float* p) { return LoadU(p); }
        FORCE_INLINE static t_i LoadU(const int* p) { return vsetq_lane_s32(*p, vdupq_n_s32(0), 0); }
        FORCE_INLINE static t_i Load(const int* p) { return LoadU(p); }
        FORCE_INLINE static void StoreU(float* p, t_f v) { *p = vgetq_lane_f32(v, 0); }
        FORCE_INLINE static void Store(float* p, t_f v) { StoreU(p, v); }
        FORCE_INLINE static void StoreU(int* p, t_i v) { *p = vgetq_lane_s32(v, 0); }
        FORCE_INLINE static void Store(int* p, t_i v) { StoreU(p, v); }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f32_u32(vceqq_f32(v1, v2));
            else
                return vreinterpretq_f32_u32(vcgtq_f32(v1, v2));
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return vreinterpretq_f32_u32(vceqq_f32(v1, v2));
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return vmulq_f32(v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return vdivq_f32(v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return vaddq_f32(v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return vsubq_f32(v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return vsqrtq_f32(v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return vfmaq_f32(v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return vfmaq_f32(vnegq_f32(v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return vfmsq_f32(v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return vminq_f32(v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return vmaxq_f32(v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return vrndmq_f32(v); }
    };

    // S_SIMDV<2,4>: load 2 floats, zero upper lanes
    template <>
    struct S_SIMDV<2, 4> : public S_NeonMultipleF {
        FORCE_INLINE static t_f LoadU(const float* p) {
            float32x2_t lo = vld1_f32(p);
            return vcombine_f32(lo, vdup_n_f32(0));
        }
        FORCE_INLINE static t_f Load(const float* p) { return LoadU(p); }
        FORCE_INLINE static t_i LoadU(const int* p) {
            int32x2_t lo = vld1_s32(p);
            return vcombine_s32(lo, vdup_n_s32(0));
        }
        FORCE_INLINE static t_i Load(const int* p) { return LoadU(p); }
        FORCE_INLINE static void StoreU(float* p, t_f v) { vst1_f32(p, vget_low_f32(v)); }
        FORCE_INLINE static void Store(float* p, t_f v) { StoreU(p, v); }
        FORCE_INLINE static void StoreU(int* p, t_i v) { vst1_s32(p, vget_low_s32(v)); }
        FORCE_INLINE static void Store(int* p, t_i v) { StoreU(p, v); }
    };

    // S_SIMDV<4,4>: full 128-bit load
    template <>
    struct S_SIMDV<4, 4> : public S_NeonMultipleF {
        FORCE_INLINE static t_f LoadU(const float* p) { return vld1q_f32(p); }
        FORCE_INLINE static t_f Load(const float* p) { return vld1q_f32(p); }
        FORCE_INLINE static t_i LoadU(const int* p) { return vld1q_s32(p); }
        FORCE_INLINE static t_i Load(const int* p) { return vld1q_s32(p); }
        FORCE_INLINE static void StoreU(float* p, t_f v) { vst1q_f32(p, v); }
        FORCE_INLINE static void Store(float* p, t_f v) { vst1q_f32(p, v); }
        FORCE_INLINE static void StoreU(int* p, t_i v) { vst1q_s32(p, v); }
        FORCE_INLINE static void Store(int* p, t_i v) { vst1q_s32(p, v); }
    };

    // S_SIMDV<8,4>: 256-bit emulated via two 128-bit float32x4_t registers
    struct S_NeonWideF {
        struct t_f { float32x4_t lo, hi; };
        struct t_i { int32x4_t lo, hi; };
        using t_base_type = float;

        FORCE_INLINE static t_f Cast(float32x4_t v) noexcept { return {v, v}; }
        FORCE_INLINE static t_i Cast(int32x4_t v) noexcept { return {v, v}; }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }
        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { auto z = vdupq_n_s32(0); return {z, z}; }
        FORCE_INLINE static t_f SetZeroF() { auto z = vdupq_n_f32(0); return {z, z}; }
        FORCE_INLINE static t_f Set1(float v) { auto x = vdupq_n_f32(v); return {x, x}; }
        FORCE_INLINE static t_i Set1(int v) { auto x = vdupq_n_s32(v); return {x, x}; }

        FORCE_INLINE static t_i CastI(t_f v) { return {vreinterpretq_s32_f32(v.lo), vreinterpretq_s32_f32(v.hi)}; }
        FORCE_INLINE static t_f CastF(t_i v) { return {vreinterpretq_f32_s32(v.lo), vreinterpretq_f32_s32(v.hi)}; }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return {vcvtq_f32_s32(v.lo), vcvtq_f32_s32(v.hi)}; }
        FORCE_INLINE static t_i CVTF2I(t_f v) { return {vcvtnq_s32_f32(v.lo), vcvtnq_s32_f32(v.hi)}; }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return {vreinterpretq_s32_u32(vceqq_s32(v1.lo, v2.lo)), vreinterpretq_s32_u32(vceqq_s32(v1.hi, v2.hi))};
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            auto shift = vdupq_n_s32(-i);
            return {vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v.lo), shift)),
                    vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v.hi), shift))};
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            auto shift = vdupq_n_s32(i);
            return {vshlq_s32(v.lo, shift), vshlq_s32(v.hi, shift)};
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            auto shift = vdupq_n_s32(-i);
            return {vshlq_s32(v.lo, shift), vshlq_s32(v.hi, shift)};
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            uint32x4_t bits_lo = vbicq_u32(vreinterpretq_u32_f32(v2.lo), vreinterpretq_u32_f32(v1.lo));
            uint32x4_t bits_hi = vbicq_u32(vreinterpretq_u32_f32(v2.hi), vreinterpretq_u32_f32(v1.hi));
            return vmaxvq_u32(vorrq_u32(bits_lo, bits_hi)) == 0;
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            uint32x4_t bits_lo = vandq_u32(vreinterpretq_u32_f32(v1.lo), vreinterpretq_u32_f32(v2.lo));
            uint32x4_t bits_hi = vandq_u32(vreinterpretq_u32_f32(v1.hi), vreinterpretq_u32_f32(v2.hi));
            return vmaxvq_u32(vorrq_u32(bits_lo, bits_hi)) == 0;
        }

        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return {vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1.lo), vreinterpretq_u32_f32(v2.lo))),
                    vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(v1.hi), vreinterpretq_u32_f32(v2.hi)))};
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return {vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v2.lo), vreinterpretq_u32_f32(v1.lo))),
                    vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(v2.hi), vreinterpretq_u32_f32(v1.hi)))};
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return {vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v1.lo), vreinterpretq_u32_f32(v2.lo))),
                    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(v1.hi), vreinterpretq_u32_f32(v2.hi)))};
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return {vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1.lo), vreinterpretq_u32_f32(v2.lo))),
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(v1.hi), vreinterpretq_u32_f32(v2.hi)))};
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return {vsubq_s32(v1.lo, v2.lo), vsubq_s32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return {vaddq_s32(v1.lo, v2.lo), vaddq_s32(v1.hi, v2.hi)}; }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            uint32x4_t mask_lo = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(v3.lo), 31));
            uint32x4_t mask_hi = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_f32(v3.hi), 31));
            return {vbslq_f32(mask_lo, v2.lo, v1.lo), vbslq_f32(mask_hi, v2.hi, v1.hi)};
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return {vreinterpretq_f32_u32(vceqq_f32(v1.lo, v2.lo)), vreinterpretq_f32_u32(vceqq_f32(v1.hi, v2.hi))};
            else
                return {vreinterpretq_f32_u32(vcgtq_f32(v1.lo, v2.lo)), vreinterpretq_f32_u32(vcgtq_f32(v1.hi, v2.hi))};
        }
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return {vreinterpretq_f32_u32(vceqq_f32(v1.lo, v2.lo)), vreinterpretq_f32_u32(vceqq_f32(v1.hi, v2.hi))};
            else
                return {vreinterpretq_f32_u32(vcgtq_f32(v1.lo, v2.lo)), vreinterpretq_f32_u32(vcgtq_f32(v1.hi, v2.hi))};
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return {vreinterpretq_f32_u32(vceqq_f32(v1.lo, v2.lo)), vreinterpretq_f32_u32(vceqq_f32(v1.hi, v2.hi))};
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return {vmulq_f32(v1.lo, v2.lo), vmulq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return {vdivq_f32(v1.lo, v2.lo), vdivq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return {vaddq_f32(v1.lo, v2.lo), vaddq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return {vsubq_f32(v1.lo, v2.lo), vsubq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Sqrt(t_f v) { return {vsqrtq_f32(v.lo), vsqrtq_f32(v.hi)}; }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return {vfmaq_f32(v3.lo, v1.lo, v2.lo), vfmaq_f32(v3.hi, v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return {vfmaq_f32(vnegq_f32(v3.lo), v1.lo, v2.lo), vfmaq_f32(vnegq_f32(v3.hi), v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return {vfmsq_f32(v3.lo, v1.lo, v2.lo), vfmsq_f32(v3.hi, v1.hi, v2.hi)}; }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return {vminq_f32(v1.lo, v2.lo), vminq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return {vmaxq_f32(v1.lo, v2.lo), vmaxq_f32(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Floor(t_f v) { return {vrndmq_f32(v.lo), vrndmq_f32(v.hi)}; }
    };

    template <>
    struct S_SIMDV<8, 4> : public S_NeonWideF {
        FORCE_INLINE static t_f LoadU(const float* p) { return {vld1q_f32(p), vld1q_f32(p + 4)}; }
        FORCE_INLINE static t_f Load(const float* p) { return LoadU(p); }
        FORCE_INLINE static t_i LoadU(const int* p) { return {vld1q_s32(p), vld1q_s32(p + 4)}; }
        FORCE_INLINE static t_i Load(const int* p) { return LoadU(p); }
        FORCE_INLINE static void StoreU(float* p, t_f v) { vst1q_f32(p, v.lo); vst1q_f32(p + 4, v.hi); }
        FORCE_INLINE static void Store(float* p, t_f v) { StoreU(p, v); }
        FORCE_INLINE static void StoreU(int* p, t_i v) { vst1q_s32(p, v.lo); vst1q_s32(p + 4, v.hi); }
        FORCE_INLINE static void Store(int* p, t_i v) { StoreU(p, v); }
    };

    //############################################################################################################################################################

    struct S_NeonBaseD {
        using t_i = int64x2_t;
        using t_f = float64x2_t;
        using t_base_type = double;

        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { return vdupq_n_s64(0); }
        FORCE_INLINE static t_f SetZeroF() { return vdupq_n_f64(0); }
        FORCE_INLINE static t_f Set1(double v) { return vdupq_n_f64(v); }
        FORCE_INLINE static t_f Set(double v1, double v2) {
            double __attribute__((aligned(16))) data[2] = {v2, v1};
            return vld1q_f64(data);
        }
        FORCE_INLINE static t_i Set1(int64_t v) { return vdupq_n_s64(v); }
        FORCE_INLINE static t_i Set(int64_t v1, int64_t v2) {
            int64_t __attribute__((aligned(16))) data[2] = {v2, v1};
            return vld1q_s64(data);
        }

        FORCE_INLINE static t_i CastI(t_f v) { return vreinterpretq_s64_f64(v); }
        FORCE_INLINE static t_f CastF(t_i v) { return vreinterpretq_f64_s64(v); }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return i64o_double(v); }
        FORCE_INLINE static t_i CVTF2I(t_f v) { return double_to_int64(v); }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return vreinterpretq_s64_u64(vceqq_s64(v1, v2));
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(v), vdupq_n_s64(-i)));
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return vshlq_s64(v, vdupq_n_s64(i));
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return vreinterpretq_s64_s32(vshlq_s32(vreinterpretq_s32_s64(v), vdupq_n_s32(-i)));
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            uint64x2_t bits = vbicq_u64(vreinterpretq_u64_f64(v2), vreinterpretq_u64_f64(v1));
            return (vgetq_lane_u64(bits, 0) | vgetq_lane_u64(bits, 1)) == 0;
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            uint64x2_t bits = vandq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2));
            return (vgetq_lane_u64(bits, 0) | vgetq_lane_u64(bits, 1)) == 0;
        }
        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)));
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v2), vreinterpretq_u64_f64(v1)));
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)));
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v1), vreinterpretq_u64_f64(v2)));
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return vsubq_s64(v1, v2); }
        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            // Broadcast sign bit to all 64 bits, matching x86 blendv semantics:
            // sign=1 (negative) selects from v2, sign=0 (non-negative) selects from v1.
            uint64x2_t mask = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(v3), 63));
            return vbslq_f64(mask, v2, v1);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f64_u64(vceqq_f64(v1, v2));
            else
                return vreinterpretq_f64_u64(vcgtq_f64(v1, v2));
        }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return vaddq_s64(v1, v2); }
    };

    struct S_NeonMultipleD : public S_NeonBaseD {
        using S_NeonBaseD::Add;
        using S_NeonBaseD::Sub;
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f64_u64(vceqq_f64(v1, v2));
            else
                return vreinterpretq_f64_u64(vcgtq_f64(v1, v2));
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(vceqq_f64(v1, v2));
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return vmulq_f64(v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return vdivq_f64(v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return vaddq_f64(v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return vsubq_f64(v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return vsqrtq_f64(v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return vfmaq_f64(v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return vfmaq_f64(vnegq_f64(v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return vfmsq_f64(v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return vminq_f64(v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return vmaxq_f64(v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return vrndmq_f64(v); }
    };

    // S_SIMDV<1,8>: load 1 double, zero upper lane
    template <>
    struct S_SIMDV<1, 8> : public S_NeonBaseD {
        using S_NeonBaseD::Add;
        using S_NeonBaseD::Sub;

        FORCE_INLINE static t_f LoadU(const double* p) { return vsetq_lane_f64(*p, vdupq_n_f64(0), 0); }
        FORCE_INLINE static t_f Load(const double* p) { return LoadU(p); }
        FORCE_INLINE static t_i LoadU(const int64_t* p) { return vsetq_lane_s64(*p, vdupq_n_s64(0), 0); }
        FORCE_INLINE static t_i Load(const int64_t* p) { return LoadU(p); }
        FORCE_INLINE static void StoreU(double* p, t_f v) { *p = vgetq_lane_f64(v, 0); }
        FORCE_INLINE static void Store(double* p, t_f v) { StoreU(p, v); }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) { *p = vgetq_lane_s64(v, 0); }
        FORCE_INLINE static void Store(int64_t* p, t_i v) { StoreU(p, v); }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return vreinterpretq_f64_u64(vceqq_f64(v1, v2));
            else
                return vreinterpretq_f64_u64(vcgtq_f64(v1, v2));
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return vreinterpretq_f64_u64(vceqq_f64(v1, v2));
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return vmulq_f64(v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return vdivq_f64(v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return vaddq_f64(v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return vsubq_f64(v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return vsqrtq_f64(v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return vfmaq_f64(v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return vfmaq_f64(vnegq_f64(v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return vfmsq_f64(v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return vminq_f64(v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return vmaxq_f64(v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return vrndmq_f64(v); }
    };

    // S_SIMDV<2,8>: full 128-bit double load
    template <>
    struct S_SIMDV<2, 8> : public S_NeonMultipleD {
        FORCE_INLINE static t_f LoadU(const double* p) { return vld1q_f64(p); }
        FORCE_INLINE static t_f Load(const double* p) { return vld1q_f64(p); }
        FORCE_INLINE static t_i LoadU(const int64_t* p) { return vld1q_s64(p); }
        FORCE_INLINE static t_i Load(const int64_t* p) { return vld1q_s64(p); }
        FORCE_INLINE static void StoreU(double* p, t_f v) { vst1q_f64(p, v); }
        FORCE_INLINE static void Store(double* p, t_f v) { vst1q_f64(p, v); }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) { vst1q_s64(p, v); }
        FORCE_INLINE static void Store(int64_t* p, t_i v) { vst1q_s64(p, v); }
    };

    // S_SIMDV<4,8>: 256-bit emulated via two 128-bit float64x2_t registers
    struct S_NeonWideD {
        struct t_f { float64x2_t lo, hi; };
        struct t_i { int64x2_t lo, hi; };
        using t_base_type = double;

        FORCE_INLINE static t_f Cast(float64x2_t v) noexcept { return {v, v}; }
        FORCE_INLINE static t_i Cast(int64x2_t v) noexcept { return {v, v}; }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }
        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { auto z = vdupq_n_s64(0); return {z, z}; }
        FORCE_INLINE static t_f SetZeroF() { auto z = vdupq_n_f64(0); return {z, z}; }
        FORCE_INLINE static t_f Set1(double v) { auto x = vdupq_n_f64(v); return {x, x}; }
        FORCE_INLINE static t_i Set1(int64_t v) { auto x = vdupq_n_s64(v); return {x, x}; }

        FORCE_INLINE static t_i CastI(t_f v) { return {vreinterpretq_s64_f64(v.lo), vreinterpretq_s64_f64(v.hi)}; }
        FORCE_INLINE static t_f CastF(t_i v) { return {vreinterpretq_f64_s64(v.lo), vreinterpretq_f64_s64(v.hi)}; }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return {i64o_double(v.lo), i64o_double(v.hi)}; }
        FORCE_INLINE static t_i CVTF2I(t_f v) { return {double_to_int64(v.lo), double_to_int64(v.hi)}; }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            return {vreinterpretq_s64_u64(vceqq_s64(v1.lo, v2.lo)), vreinterpretq_s64_u64(vceqq_s64(v1.hi, v2.hi))};
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            auto shift = vdupq_n_s64(-i);
            return {vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(v.lo), shift)),
                    vreinterpretq_s64_u64(vshlq_u64(vreinterpretq_u64_s64(v.hi), shift))};
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            auto shift = vdupq_n_s64(i);
            return {vshlq_s64(v.lo, shift), vshlq_s64(v.hi, shift)};
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            auto shift = vdupq_n_s32(-i);
            return {vreinterpretq_s64_s32(vshlq_s32(vreinterpretq_s32_s64(v.lo), shift)),
                    vreinterpretq_s64_s32(vshlq_s32(vreinterpretq_s32_s64(v.hi), shift))};
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            uint64x2_t bits_lo = vbicq_u64(vreinterpretq_u64_f64(v2.lo), vreinterpretq_u64_f64(v1.lo));
            uint64x2_t bits_hi = vbicq_u64(vreinterpretq_u64_f64(v2.hi), vreinterpretq_u64_f64(v1.hi));
            uint64x2_t combined = vorrq_u64(bits_lo, bits_hi);
            return (vgetq_lane_u64(combined, 0) | vgetq_lane_u64(combined, 1)) == 0;
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            uint64x2_t bits_lo = vandq_u64(vreinterpretq_u64_f64(v1.lo), vreinterpretq_u64_f64(v2.lo));
            uint64x2_t bits_hi = vandq_u64(vreinterpretq_u64_f64(v1.hi), vreinterpretq_u64_f64(v2.hi));
            uint64x2_t combined = vorrq_u64(bits_lo, bits_hi);
            return (vgetq_lane_u64(combined, 0) | vgetq_lane_u64(combined, 1)) == 0;
        }

        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return {vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v1.lo), vreinterpretq_u64_f64(v2.lo))),
                    vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v1.hi), vreinterpretq_u64_f64(v2.hi)))};
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return {vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v2.lo), vreinterpretq_u64_f64(v1.lo))),
                    vreinterpretq_f64_u64(vbicq_u64(vreinterpretq_u64_f64(v2.hi), vreinterpretq_u64_f64(v1.hi)))};
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return {vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(v1.lo), vreinterpretq_u64_f64(v2.lo))),
                    vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(v1.hi), vreinterpretq_u64_f64(v2.hi)))};
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return {vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v1.lo), vreinterpretq_u64_f64(v2.lo))),
                    vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(v1.hi), vreinterpretq_u64_f64(v2.hi)))};
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return {vsubq_s64(v1.lo, v2.lo), vsubq_s64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return {vaddq_s64(v1.lo, v2.lo), vaddq_s64(v1.hi, v2.hi)}; }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            uint64x2_t mask_lo = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(v3.lo), 63));
            uint64x2_t mask_hi = vreinterpretq_u64_s64(vshrq_n_s64(vreinterpretq_s64_f64(v3.hi), 63));
            return {vbslq_f64(mask_lo, v2.lo, v1.lo), vbslq_f64(mask_hi, v2.hi, v1.hi)};
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return {vreinterpretq_f64_u64(vceqq_f64(v1.lo, v2.lo)), vreinterpretq_f64_u64(vceqq_f64(v1.hi, v2.hi))};
            else
                return {vreinterpretq_f64_u64(vcgtq_f64(v1.lo, v2.lo)), vreinterpretq_f64_u64(vcgtq_f64(v1.hi, v2.hi))};
        }
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ)
                return {vreinterpretq_f64_u64(vceqq_f64(v1.lo, v2.lo)), vreinterpretq_f64_u64(vceqq_f64(v1.hi, v2.hi))};
            else
                return {vreinterpretq_f64_u64(vcgtq_f64(v1.lo, v2.lo)), vreinterpretq_f64_u64(vcgtq_f64(v1.hi, v2.hi))};
        }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) {
            return {vreinterpretq_f64_u64(vceqq_f64(v1.lo, v2.lo)), vreinterpretq_f64_u64(vceqq_f64(v1.hi, v2.hi))};
        }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return {vmulq_f64(v1.lo, v2.lo), vmulq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return {vdivq_f64(v1.lo, v2.lo), vdivq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return {vaddq_f64(v1.lo, v2.lo), vaddq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return {vsubq_f64(v1.lo, v2.lo), vsubq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Sqrt(t_f v) { return {vsqrtq_f64(v.lo), vsqrtq_f64(v.hi)}; }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return {vfmaq_f64(v3.lo, v1.lo, v2.lo), vfmaq_f64(v3.hi, v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return {vfmaq_f64(vnegq_f64(v3.lo), v1.lo, v2.lo), vfmaq_f64(vnegq_f64(v3.hi), v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return {vfmsq_f64(v3.lo, v1.lo, v2.lo), vfmsq_f64(v3.hi, v1.hi, v2.hi)}; }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return {vminq_f64(v1.lo, v2.lo), vminq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return {vmaxq_f64(v1.lo, v2.lo), vmaxq_f64(v1.hi, v2.hi)}; }
        FORCE_INLINE static t_f Floor(t_f v) { return {vrndmq_f64(v.lo), vrndmq_f64(v.hi)}; }
    };

    template <>
    struct S_SIMDV<4, 8> : public S_NeonWideD {
        FORCE_INLINE static t_f LoadU(const double* p) { return {vld1q_f64(p), vld1q_f64(p + 2)}; }
        FORCE_INLINE static t_f Load(const double* p) { return LoadU(p); }
        FORCE_INLINE static t_i LoadU(const int64_t* p) { return {vld1q_s64(p), vld1q_s64(p + 2)}; }
        FORCE_INLINE static t_i Load(const int64_t* p) { return LoadU(p); }
        FORCE_INLINE static void StoreU(double* p, t_f v) { vst1q_f64(p, v.lo); vst1q_f64(p + 2, v.hi); }
        FORCE_INLINE static void Store(double* p, t_f v) { StoreU(p, v); }
        FORCE_INLINE static void StoreU(int64_t* p, t_i v) { vst1q_s64(p, v.lo); vst1q_s64(p + 2, v.hi); }
        FORCE_INLINE static void Store(int64_t* p, t_i v) { StoreU(p, v); }
    };

    // Emulated 256-bit width: 32 bytes
    template <size_t I_ElemSize>
    using S_MaxSIMD = S_SIMDV<32 / I_ElemSize, I_ElemSize>;

    //############################################################################################################################################################
    // SVE specializations: N=0 is a VLA tag (no existing specialization uses N=0).
    // Only available when compiled with -march=armv8-a+sve (or similar).
    // Kept in SIMDFunctions.h (not a separate header) for consistency with
    // the x86/NEON specializations above. Only ops_sve.cpp compiles with
    // SVE enabled, so these specializations are confined to that TU.
    //############################################################################################################################################################
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>

    // S_SIMDV<0,4>: SVE vector-length-agnostic float32
    template <>
    struct S_SIMDV<0, 4> {
        using t_f = svfloat32_t;
        using t_i = svint32_t;
        using t_base_type = float;

        // Cast from NEON constant to SVE by extracting lane 0 and broadcasting.
        // INVARIANT: all S_Constants values are produced by Set1 (uniform broadcast),
        // so lane 0 == all lanes. If a non-broadcast constant is ever added, these
        // casts will silently produce incorrect results for SVE.
        FORCE_INLINE static t_f Cast(float32x4_t v) noexcept { return svdup_n_f32(vgetq_lane_f32(v, 0)); }
        FORCE_INLINE static t_i Cast(int32x4_t v) noexcept { return svdup_n_s32(vgetq_lane_s32(v, 0)); }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }
        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { return svdup_n_s32(0); }
        FORCE_INLINE static t_f SetZeroF() { return svdup_n_f32(0); }
        FORCE_INLINE static t_f Set1(float v) { return svdup_n_f32(v); }
        FORCE_INLINE static t_i Set1(int v) { return svdup_n_s32(v); }

        FORCE_INLINE static t_i CastI(t_f v) { return svreinterpret_s32_f32(v); }
        FORCE_INLINE static t_f CastF(t_i v) { return svreinterpret_f32_s32(v); }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return svcvt_f32_s32_x(svptrue_b32(), v); }
        // Round to nearest-even explicitly before converting, matching NEON
        // vcvtnq / x86 cvtps2dq semantics regardless of FPCR rounding mode.
        FORCE_INLINE static t_i CVTF2I(t_f v) { return svcvt_s32_f32_x(svptrue_b32(), svrintn_f32_x(svptrue_b32(), v)); }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            svbool_t pred = svcmpeq_s32(svptrue_b32(), v1, v2);
            return svsel_s32(pred, svdup_n_s32(-1), svdup_n_s32(0));
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return svreinterpret_s32_u32(svlsr_u32_x(svptrue_b32(), svreinterpret_u32_s32(v), svdup_n_u32((uint32_t)i)));
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return svlsl_s32_x(svptrue_b32(), v, svdup_n_u32((uint32_t)i));
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            return svasr_s32_x(svptrue_b32(), v, svdup_n_u32((uint32_t)i));
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            svuint32_t bits = svbic_u32_x(svptrue_b32(), svreinterpret_u32_f32(v2), svreinterpret_u32_f32(v1));
            return !svptest_any(svptrue_b32(), svcmpne_n_u32(svptrue_b32(), bits, 0));
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            svuint32_t bits = svand_u32_x(svptrue_b32(), svreinterpret_u32_f32(v1), svreinterpret_u32_f32(v2));
            return !svptest_any(svptrue_b32(), svcmpne_n_u32(svptrue_b32(), bits, 0));
        }

        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return svreinterpret_f32_u32(svand_u32_x(svptrue_b32(), svreinterpret_u32_f32(v1), svreinterpret_u32_f32(v2)));
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return svreinterpret_f32_u32(svbic_u32_x(svptrue_b32(), svreinterpret_u32_f32(v2), svreinterpret_u32_f32(v1)));
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return svreinterpret_f32_u32(svorr_u32_x(svptrue_b32(), svreinterpret_u32_f32(v1), svreinterpret_u32_f32(v2)));
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return svreinterpret_f32_u32(sveor_u32_x(svptrue_b32(), svreinterpret_u32_f32(v1), svreinterpret_u32_f32(v2)));
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return svsub_s32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return svadd_s32_x(svptrue_b32(), v1, v2); }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            svbool_t mask = svcmplt_s32(svptrue_b32(), svreinterpret_s32_f32(v3), svdup_n_s32(0));
            return svsel_f32(mask, v2, v1);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ) {
                svbool_t pred = svcmpeq_f32(svptrue_b32(), v1, v2);
                return svreinterpret_f32_u32(svsel_u32(pred, svdup_n_u32(~(uint32_t)0), svdup_n_u32(0)));
            } else {
                svbool_t pred = svcmpgt_f32(svptrue_b32(), v1, v2);
                return svreinterpret_f32_u32(svsel_u32(pred, svdup_n_u32(~(uint32_t)0), svdup_n_u32(0)));
            }
        }
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) { return CmpFM<I_Mode>(v1, v2); }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) { return CmpFM<_CMP_EQ_OQ>(v1, v2); }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return svmul_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return svdiv_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return svadd_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return svsub_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return svsqrt_f32_x(svptrue_b32(), v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return svmla_f32_x(svptrue_b32(), v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return svmla_f32_x(svptrue_b32(), svneg_f32_x(svptrue_b32(), v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return svmls_f32_x(svptrue_b32(), v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return svmin_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return svmax_f32_x(svptrue_b32(), v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return svrintm_f32_x(svptrue_b32(), v); }

        FORCE_INLINE static t_f LoadU(const float* p) { return svld1_f32(svptrue_b32(), p); }
        FORCE_INLINE static t_f Load(const float* p) { return svld1_f32(svptrue_b32(), p); }
        FORCE_INLINE static void StoreU(float* p, t_f v) { svst1_f32(svptrue_b32(), p, v); }
        FORCE_INLINE static void Store(float* p, t_f v) { svst1_f32(svptrue_b32(), p, v); }
    };

    // S_SIMDV<0,8>: SVE vector-length-agnostic float64
    template <>
    struct S_SIMDV<0, 8> {
        using t_f = svfloat64_t;
        using t_i = svint64_t;
        using t_base_type = double;

        // See S_SIMDV<0, 4>::Cast comment — same broadcast invariant applies.
        FORCE_INLINE static t_f Cast(float64x2_t v) noexcept { return svdup_n_f64(vgetq_lane_f64(v, 0)); }
        FORCE_INLINE static t_i Cast(int64x2_t v) noexcept { return svdup_n_s64(vgetq_lane_s64(v, 0)); }
        FORCE_INLINE static t_f Cast(t_f v) noexcept { return v; }
        FORCE_INLINE static t_i Cast(t_i v) noexcept { return v; }

        FORCE_INLINE static t_i SetZeroI() { return svdup_n_s64(0); }
        FORCE_INLINE static t_f SetZeroF() { return svdup_n_f64(0); }
        FORCE_INLINE static t_f Set1(double v) { return svdup_n_f64(v); }
        FORCE_INLINE static t_i Set1(int64_t v) { return svdup_n_s64(v); }

        FORCE_INLINE static t_i CastI(t_f v) { return svreinterpret_s64_f64(v); }
        FORCE_INLINE static t_f CastF(t_i v) { return svreinterpret_f64_s64(v); }
        FORCE_INLINE static t_f CVTI2F(t_i v) { return svcvt_f64_s64_x(svptrue_b64(), v); }
        // Round to nearest-even explicitly before converting, matching NEON
        // vcvtnq / x86 cvtps2dq semantics regardless of FPCR rounding mode.
        FORCE_INLINE static t_i CVTF2I(t_f v) { return svcvt_s64_f64_x(svptrue_b64(), svrintn_f64_x(svptrue_b64(), v)); }

        FORCE_INLINE static t_i CmpEqI(t_i v1, t_i v2) {
            svbool_t pred = svcmpeq_s64(svptrue_b64(), v1, v2);
            return svsel_s64(pred, svdup_n_s64(-1), svdup_n_s64(0));
        }
        FORCE_INLINE static t_i SRLI(t_i v, int i) {
            return svreinterpret_s64_u64(svlsr_u64_x(svptrue_b64(), svreinterpret_u64_s64(v), svdup_n_u64((uint64_t)i)));
        }
        FORCE_INLINE static t_i SLLI(t_i v, int i) {
            return svlsl_s64_x(svptrue_b64(), v, svdup_n_u64((uint64_t)i));
        }
        FORCE_INLINE static t_i SRAI32(t_i v, int i) {
            svint32_t v32 = svreinterpret_s32_s64(v);
            v32 = svasr_s32_x(svptrue_b32(), v32, svdup_n_u32((uint32_t)i));
            return svreinterpret_s64_s32(v32);
        }

        FORCE_INLINE static int TestCF(t_f v1, t_f v2) {
            svuint64_t bits = svbic_u64_x(svptrue_b64(), svreinterpret_u64_f64(v2), svreinterpret_u64_f64(v1));
            return !svptest_any(svptrue_b64(), svcmpne_n_u64(svptrue_b64(), bits, 0));
        }
        FORCE_INLINE static int TestZF(t_f v1, t_f v2) {
            svuint64_t bits = svand_u64_x(svptrue_b64(), svreinterpret_u64_f64(v1), svreinterpret_u64_f64(v2));
            return !svptest_any(svptrue_b64(), svcmpne_n_u64(svptrue_b64(), bits, 0));
        }

        FORCE_INLINE static t_f AndF(t_f v1, t_f v2) {
            return svreinterpret_f64_u64(svand_u64_x(svptrue_b64(), svreinterpret_u64_f64(v1), svreinterpret_u64_f64(v2)));
        }
        FORCE_INLINE static t_f AndNotF(t_f v1, t_f v2) {
            return svreinterpret_f64_u64(svbic_u64_x(svptrue_b64(), svreinterpret_u64_f64(v2), svreinterpret_u64_f64(v1)));
        }
        FORCE_INLINE static t_f OrF(t_f v1, t_f v2) {
            return svreinterpret_f64_u64(svorr_u64_x(svptrue_b64(), svreinterpret_u64_f64(v1), svreinterpret_u64_f64(v2)));
        }
        FORCE_INLINE static t_f XorF(t_f v1, t_f v2) {
            return svreinterpret_f64_u64(sveor_u64_x(svptrue_b64(), svreinterpret_u64_f64(v1), svreinterpret_u64_f64(v2)));
        }

        FORCE_INLINE static t_i Sub(t_i v1, t_i v2) { return svsub_s64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_i Add(t_i v1, t_i v2) { return svadd_s64_x(svptrue_b64(), v1, v2); }

        FORCE_INLINE static t_f BlendVF(t_f v1, t_f v2, t_f v3) {
            svbool_t mask = svcmplt_s64(svptrue_b64(), svreinterpret_s64_f64(v3), svdup_n_s64(0));
            return svsel_f64(mask, v2, v1);
        }

        template <int I_Mode>
        FORCE_INLINE static t_f CmpFM(t_f v1, t_f v2) {
            if constexpr (I_Mode == _CMP_EQ_OQ) {
                svbool_t pred = svcmpeq_f64(svptrue_b64(), v1, v2);
                return svreinterpret_f64_u64(svsel_u64(pred, svdup_n_u64(~(uint64_t)0), svdup_n_u64(0)));
            } else {
                svbool_t pred = svcmpgt_f64(svptrue_b64(), v1, v2);
                return svreinterpret_f64_u64(svsel_u64(pred, svdup_n_u64(~(uint64_t)0), svdup_n_u64(0)));
            }
        }
        template <int I_Mode>
        FORCE_INLINE static t_f CmpF(t_f v1, t_f v2) { return CmpFM<I_Mode>(v1, v2); }
        FORCE_INLINE static t_f CmpEqF(t_f v1, t_f v2) { return CmpFM<_CMP_EQ_OQ>(v1, v2); }

        FORCE_INLINE static t_f Mul(t_f v1, t_f v2) { return svmul_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Div(t_f v1, t_f v2) { return svdiv_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Add(t_f v1, t_f v2) { return svadd_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Sub(t_f v1, t_f v2) { return svsub_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Sqrt(t_f v) { return svsqrt_f64_x(svptrue_b64(), v); }

        FORCE_INLINE static t_f FMADD(t_f v1, t_f v2, t_f v3) { return svmla_f64_x(svptrue_b64(), v3, v1, v2); }
        FORCE_INLINE static t_f FMSUB(t_f v1, t_f v2, t_f v3) { return svmla_f64_x(svptrue_b64(), svneg_f64_x(svptrue_b64(), v3), v1, v2); }
        FORCE_INLINE static t_f FNMADD(t_f v1, t_f v2, t_f v3) { return svmls_f64_x(svptrue_b64(), v3, v1, v2); }

        FORCE_INLINE static t_f Min(t_f v1, t_f v2) { return svmin_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Max(t_f v1, t_f v2) { return svmax_f64_x(svptrue_b64(), v1, v2); }
        FORCE_INLINE static t_f Floor(t_f v) { return svrintm_f64_x(svptrue_b64(), v); }

        FORCE_INLINE static t_f LoadU(const double* p) { return svld1_f64(svptrue_b64(), p); }
        FORCE_INLINE static t_f Load(const double* p) { return svld1_f64(svptrue_b64(), p); }
        FORCE_INLINE static void StoreU(double* p, t_f v) { svst1_f64(svptrue_b64(), p, v); }
        FORCE_INLINE static void Store(double* p, t_f v) { svst1_f64(svptrue_b64(), p, v); }

        // Set with individual values — only used in LnV for N==1 and N==2 branches,
        // which are not taken for SVE (N==0). Must exist to satisfy compilation but
        // must never actually be called.
        FORCE_INLINE static t_f Set(double, double) { __builtin_unreachable(); }
    };

#endif // __ARM_FEATURE_SVE
}

#else
#error "Unsupported architecture: fastops requires x86 (SSE/AVX) or AArch64 (NEON)"
#endif

#endif // SIMD_FUNCTIONS_H
