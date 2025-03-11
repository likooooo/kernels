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
            T phase_term = 2_PI * freq * (k.real() * x /xsize + k.imag() * y / ysize);
            *p = complex_t<T>(std::cos(phase_term), std::sin(phase_term));
            p++;
        });
    }
    template<class T, size_t N>inline void phase_modulate(
        complex_t<T>* p, const std::array<size_t, N>& shape, const std::array<T, N>& shift_pixel)
    {
        complex_t<T> TWOPI = complex_t<T>(0, -2_PI);
        center_zero_loop_distance_2<T, N>(shape, default_step<T, N>(1), 
            [&](const std::array<T, N>& index, T kr){
                T sum = 0;
                for(size_t i = 0; i < N; i++){
                    sum += shift_pixel[i] * index[i] / shape[i]; 
                }
                complex_t<T> phase = TWOPI * sum; 
                *p = std::exp(phase);
                p++;
            }
        );
    }
    template<class T, size_t N>inline void free_propagation(
        complex_t<T>* p, const std::array<size_t, N>& shape, const std::array<T, N>& step, 
        T freq, T dist, complex_t<T> nk = complex_t<T>(1))
    {
        const complex_t<T> dielectric = nk * nk;
        complex_t<T> rDZTWOPI = complex_t<T>(0, 1) * T(2_PI) * freq * dist;
        center_zero_loop_distance_2<T, N>(shape, step, 
            [&](const std::array<T, N>& pos, T kr){
                complex_t<T> phase = rDZTWOPI * std::sqrt(dielectric - kr); 
                *p = std::exp(phase);
                p++;
            }
        );
    }

}