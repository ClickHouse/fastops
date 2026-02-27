vector operations library
=================================

This small library enables acceleration of bulk calls of certain math functions using SIMD instructions. Currently supported operations are exp, exp2, exp10, log, sigmoid and tanh. The library is designed with extensibility in mind. Optimized helper functions are found in `fastops/core/FastIntrinsics.h` and you are welcome to contribute your own.

Supported architectures:
* **x86 (SSE/AVX/AVX2)**: runtime CPU dispatch selects the best available instruction set. Pre-AVX fallback uses the fmath library.
* **AArch64 (NEON)**: 128-bit NEON SIMD with native FMA. Always enabled on AArch64 — no runtime dispatch needed.

`fastops/fastops.h` header provides the interface for the best version of each function. All functions are approximate, yet quite precise. Accuracy of each operation is detailed below along with operation description. All implementation architectures share the same polynomial coefficients and evaluation scheme, so accuracy is consistent across platforms.

Core implementation (`fastops/core/FastIntrinsics.h`) contains versions for fixed-size arrays that produce completely unrolled code for uncompromized performance. These may slowdown compilation and thus are currently not exposed via high-level dispatched interfaces. Be careful when using these versions: long fixed-size arrays may lead to etxreme code bloat. The regular versions perform on par with these ones if your arrays are larger than 512 bytes. 

The quote from Mikhail Parakhin, the Yandex CTO and library creator:
//In its spirit the library is aimed to aid vectorization of any compute. Just code up the performer class similar to the existing ones and let it fly. Current performers include not only compute, but also memset/memcopy/memmove operations that are always inlined and so work much faster for short arrays. On short strings gain may be as high as 2x.//

How to build
=================================
The library requires C++ compiler with c++17 support. The library itslef only depends on fmath, which is single-header library. The supporting code - `eval` and `benchmark` programs depend on TCLAP options parsing library. These dependencies are placed into contrib/libs inside root fastops directory.

The library is built using cmake and the build precess is simple and straighforward:
1. $ mkdir build install
2. $ cd build
3. $ cmake ../
4. $ make
5. $ make install

Tools
=================================
Two tools are provided along the library:
* tools/eval - let one check the accuracy of operations under different conditions.
* tools/benchmark - compares performance of AVX/AVX2 optimized versions with baseline fmath implementation.
Use `--help` for set of supported options.

On x86, please note that running these tools on pre-AVX hardware makes little sense.
* tools/benchmark will refuse to run on it and won't call AVX2 functions on pure AVX hardware.
* tools/eval allows selecting instruction set via command line and does not perform any checks.
  It will just crash if ran on incompatible hardware.

On AArch64, both tools use the NEON implementation directly.

Functions
=================================
The dispathed interfaces are available via `fastops/fastops.h` header file. There are single and
double precision versions for each operation. Template parameters include speed/accuracy and alignment controls.
* Speed/accuracy is bool letting you choose faster or more precise version of algorithm.
* Alignment control allows to select whether the output array is aligned or not
  (32-byte on x86 AVX, 16-byte on AArch64 NEON). The common belief is that unaligned
  versions may perform slower, but special studies for our functions were not performed.
  Choose this parameter according to your array alignment: aligned SIMD operations on
  unaligned data may crash.

All the library functionality is directly available via `fastops/core/FastIntrinsics.h` header, but then you should care about hardware compatibility yourself. On x86, a tiny AVX and AVX2 hardware detection utility is available via `fastops/core/avx_id.h`. On AArch64, NEON is always present so no detection is needed.

Below we use the following terms:
<UL> * x - input value </UL>
<UL> * EPS - relative error: EPS = abs(approx - real) / (abs(real) + 1e-100);</UL>

## Exp
Compute exponent function.
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Exp(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Exp(const double* from, size_t size, double* to);
```

### Accuracy by version
1. float, inexact:
  <UL> * x < -87: accuracy degrades sharply, exp(x) <= 1.0001 * true_exp(x), usually significantly less. This is due to saturation of the single precision range in inexact version. If denormals are banned the true_exp() will exhibit the same behavior.</UL>
  <UL> * x >= -87: EPS <= 7.21e-06</UL>
2. float, exact:
  <UL> * x < -87: for the most cases result is accurate. The corner cases are observed only due to different rounding directions of true_exp() and our imlementation. The results may differ up to 2x, but this is acceptable in denornals: the results are quite approximate in any case, these are just slightly different approximations.</UL>
  <UL> * x >= -87: EPS <= 4e-06</UL>
3. double, inexact
  <UL> * x < -708.39: exp(x) <= 1.0001 * true_exp(x), usually significantly less.</UL>
  <UL> * x >= -708.39: EPS <= 3.5e-06</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 2.3e-9</UL>

## Exp2
Compute base-2 exponent function: exp2(x) = 2^x. Internally this is a direct call to the Pow2V kernel with no input scaling, so it is slightly faster than `Exp`.
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Exp2(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Exp2(const double* from, size_t size, double* to);
```

### Accuracy by version
Same polynomial evaluation as `Exp`, so accuracy characteristics are identical — only the input saturation boundaries differ.
1. float, inexact:
  <UL> * x < -125: accuracy degrades sharply due to saturation of the single precision range.</UL>
  <UL> * x >= -125: EPS <= 7.21e-06</UL>
2. float, exact:
  <UL> * x < -126: corner cases near denormals, same as `Exp`.</UL>
  <UL> * x >= -126: EPS <= 4e-06</UL>
3. double, inexact
  <UL> * x < -1020: accuracy degrades sharply.</UL>
  <UL> * x >= -1020: EPS <= 3.5e-06</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 2.3e-9</UL>

## Exp10
Compute base-10 exponent function: exp10(x) = 10^x. Internally this is Pow2V(x * log2(10)), sharing the same kernel as `Exp`.
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Exp10(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Exp10(const double* from, size_t size, double* to);
```

### Accuracy by version
Same polynomial evaluation as `Exp`, but the extra multiply by log2(10) adds a small additional error.
1. float, inexact:
  <UL> * x < -37.5: accuracy degrades sharply due to saturation of the single precision range.</UL>
  <UL> * x >= -37.5: EPS <= 8e-06</UL>
2. float, exact:
  <UL> * x < -38: corner cases near denormals, same as `Exp`.</UL>
  <UL> * x >= -38: EPS <= 5e-06</UL>
3. double, inexact
  <UL> * x < -307: accuracy degrades sharply.</UL>
  <UL> * x >= -307: EPS <= 4e-06</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 3e-9</UL>

## Log
Computes natural log function.
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Log(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Log(const double* from, size_t size, double* to);
```

### Accuracy by version
1. float, inexact:
  <UL> * x < 1.17613e-38: the result almost stops decreasing around the value of -88, while actual log function still does. This leads result to become significantly greater than actual value. If denormals support is disabled the function will return -inf same as precise one.</UL>
  <UL> * x >= 1.17613e-38: EPS <= 1e-5</UL>
2. float, exact:
  <UL> * Entire range: EPS <= 4e-7</UL>
3. double, inexact
  <UL> * x < 2.99279772e-308: the result almost stops decreasing around the value of -708, while actual log function still does. This leads result to become significantly greater than actual value.</UL>
  <UL> * x >= 2.99279772e-308: EPS <= 1e-5</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 2e-7</UL>

## Sigmoid
Computes sigmoid function: sigm(x) = 1.0 / (1.0 + exp(-x)).
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Sigmoid(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Sigmoid(const double* from, size_t size, double* to);
```

### Accuracy by version
1. float, inexact:
  <UL> * Entire range: EPS <= 8e-6</UL>
  <UL> * x >= 1.17613e-38: EPS <= 1e-5</UL>
2. float, exact:
  <UL> * Entire range: EPS <= 4.5e-6</UL>
3. double, inexact
  <UL> * Entire range: EPS <= 4e-6</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 1e-12</UL>

## Tanh
Computes hyperbolic tangent (tanh) function
```
template <bool I_Exact=false, bool I_OutAligned=false>
void Tanh(const float* from, size_t size, float* to);

template <bool I_Exact=false, bool I_OutAligned=false>
void Tanh(const double* from, size_t size, double* to);
```

### Accuracy by version
Due to behavior of tanh around 0 the reative error there is unstable. The computational algorithm is close to sigmoid, so this instability is related to error measuring rather than to function computation. Due to this reason absolute error may provide more adequate indication of accuracy than relative one in some cases.

1. float, inexact:
  <UL> * [-1, 1]: maximal absolute error is around 1e-06</UL>
  <UL> * Outside [-1, 1]: EPS < 1.1e-06</UL>
2. float, exact:
  <UL> * [-1, 1]: maximal absolute error is around 1.e-06</UL>
  <UL> * Outside [-1, 1]: EPS < 2.5e-07</UL>
3. double, inexact
  <UL> * Outside [-1, 1]: EPS < 1e-06</UL>
  <UL> * [-1, -0.1] V [0.1, 1]: EPS < 1.6e-05</UL>
  <UL> * [-0.1, -0.01] V [0.01, 0.1]: EPS < 1.2e-04</UL>
  <UL> * [-0.01, 0.01]: EPS <= 3e-04 or maximal absolute error is 1e-06</UL>
4. double, exact:
  <UL> * Entire range: EPS <= 1e-11</UL>
