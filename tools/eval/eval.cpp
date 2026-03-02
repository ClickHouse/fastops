#include "opts.h"

#include <fastops/fastops.h>
#if !(defined(__aarch64__) || defined(__arm__))
#include <fastops/avx/ops_avx.h>
#include <fastops/avx2/ops_avx2.h>
#endif
#include <fastops/plain/ops_plain.h>

#include <iostream>

#if !(defined(__aarch64__) || defined(__arm__))
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif

#include <math.h>

static double GetRelError(double real, double approx) {
    return fabs(approx - real) / (fabs(real) + 1e-100);
}

int main(int argc, char** argv) {
#if defined(__aarch64__) || defined(__arm__)
    std::cerr << "Eval tool is not supported on this architecture." << std::endl;
    (void)argc; (void)argv;
    return 0;
#else
    TEvalOpts opts = ParseOptions(argc, argv);

    if (opts.NoDenormals) {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    if (!(opts.Arch == "plain" || opts.Arch == "avx2" || opts.Arch == "avx")) {
        std::cerr << "requirement arch == \"plain\" || arch == \"avx2\" || arch == \"avx\" failed" << std::endl;
        exit(1);
    }
    if (!(opts.Func == "exp" || opts.Func == "exp2" || opts.Func == "exp10" || opts.Func == "log" || opts.Func == "sigm" || opts.Func == "tanh")) {
        std::cerr << "requirement func == \"exp\" || func == \"exp2\" || func == \"exp10\" || func == \"log\" || func == \"sigm\" || func == \"tanh\" failed" << std::endl;
        exit(1);
    }

    bool isAvx2 = opts.Arch == "avx2";
    bool isAvx = opts.Arch == "avx";
    float lo = opts.Lo;
    float hi = opts.Hi;
    size_t n = opts.N;

    if (opts.Print) {
        std::cout << "input\ttrue\tapprox" << std::endl;
    }

    double maxError = 0;
    if (opts.Func == "exp") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                float trueVal = exp(input);
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::ExpAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::ExpAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::ExpPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::ExpAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::ExpAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::ExpPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                double trueVal = exp(input);
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::ExpAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::ExpAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::ExpPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::ExpAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::ExpAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::ExpPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    } else if (opts.Func == "exp2") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                float trueVal = exp2(input);
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::Exp2Avx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp2Avx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp2Plain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::Exp2Avx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp2Avx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp2Plain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                double trueVal = exp2(input);
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::Exp2Avx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp2Avx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp2Plain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::Exp2Avx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp2Avx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp2Plain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    } else if (opts.Func == "exp10") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                float trueVal = pow(10.0f, input);
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::Exp10Avx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp10Avx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp10Plain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::Exp10Avx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp10Avx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp10Plain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                double trueVal = pow(10.0, input);
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::Exp10Avx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp10Avx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp10Plain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::Exp10Avx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::Exp10Avx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::Exp10Plain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    } else if (opts.Func == "log") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                input = exp(input);
                float trueVal = log(input);
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::LogAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::LogAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::LogPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::LogAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::LogAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::LogPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                input = exp(input);
                double trueVal = log(input);
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::LogAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::LogAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::LogPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::LogAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::LogAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::LogPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    } else if (opts.Func == "sigm") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                float trueVal = 1.0 / (1.0 + exp(-input));
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::SigmoidAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::SigmoidAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::SigmoidPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::SigmoidAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::SigmoidAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::SigmoidPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                double trueVal = 1.0 / (1.0 + exp(-input));
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::SigmoidAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::SigmoidAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::SigmoidPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::SigmoidAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::SigmoidAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::SigmoidPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    } else if (opts.Func == "tanh") {
        for (size_t i = 0; i < n + 1; ++i) {
            double error = 0;
            if (!opts.UseDouble) {
                float input = lo + (hi - lo) * i * 1.0 / n;
                float trueVal = tanh(input);
                float approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::TanhAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::TanhAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::TanhPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::TanhAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::TanhAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::TanhPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            } else {
                double input = lo + (hi - lo) * i * 1.0 / n;
                double trueVal = tanh(input);
                double approxVal;
                if (opts.Exact) {
                    if (isAvx2) {
                        NFastOps::TanhAvx2<true, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::TanhAvx<true, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::TanhPlain(&input, 1, &approxVal);
                    }
                } else {
                    if (isAvx2) {
                        NFastOps::TanhAvx2<false, false>(&input, 1, &approxVal);
                    } else if (isAvx) {
                        NFastOps::TanhAvx<false, false>(&input, 1, &approxVal);
                    } else {
                        NFastOps::TanhPlain(&input, 1, &approxVal);
                    }
                }
                error = GetRelError(trueVal, approxVal);
                if (opts.Print) {
                    std::cout << input << "\t" << trueVal << "\t" << approxVal << "\t" << error << std::endl;
                }
            }
            if (error > maxError) {
                maxError = error;
            }
        }
    }
    std::cout << "Maximum relative error: " << maxError << std::endl;
#endif
}
