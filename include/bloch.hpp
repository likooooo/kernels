#pragma once
#include "kernel_loop.hpp"

namespace kernels
{
    template<class T, size_t N> inline void bloch_phase(complex_t<T>* p, const std::array<size_t, N>& shape,T freq,  T crao, T azimuth)
    {
        static_assert(N == 2);
        complex_t<T> k =  std::sin(crao) * std::exp(complex_t<T>(0, 1) * azimuth);
        const auto [ysize, xsize] = shape;
        kernel_loop<T, N>(shape, [&](const std::array<T, N>& center, const std::array<size_t, N>& indices) {
            const auto [y, x] = indices;
            T phase_term = 2 * M_PI * freq * (k.real() * x /xsize + k.imag() * y / ysize);
            *p = complex_t<T>(std::cos(phase_term), std::sin(phase_term));
            p++;
        });
    }
}