#pragma once
#include "kernel_loop.hpp"

namespace kernels
{
    template<class T, size_t N> inline void bloch_phase(complex_t<T>* p, const std::array<size_t, N>& shape, T freq,  vec2<T> k)
    {
        static_assert(N == 2);
        const auto [ysize, xsize] = shape;
        kernel_loop<T, N>(shape, [&](const std::array<T, N>& useless_center, const std::array<size_t, N>& indices) {
        // center_zero_loop_square_r<T, N>(shape, default_step<T, N>(1), 
        //     [&](const std::array<T, N>& indices, T kr){
            const auto [y, x] = indices;
            complex_t<T> TWOPI = 2_PI_I;
            *p = std::exp(TWOPI * (freq * (k.at(1) * x /xsize + k.at(0) * y / ysize)));
            p++;
        });
    }
    template<class T, size_t N>inline void phase_modulate(
        complex_t<T>* p, const std::array<size_t, N>& shape, const std::array<T, N>& shift_pixel)
    {
        complex_t<T> TWOPI = 2_PI_I;
        center_zero_loop_square_r<T, N>(shape, default_step<T, N>(1), 
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
        complex_t<T> rDZTWOPI = complex_t<T>(2_PI_I) * freq * dist;
        center_zero_loop_square_r<T, N>(shape, step, 
            [&](const std::array<T, N>& pos, T kr){
                complex_t<T> phase = rDZTWOPI * std::sqrt(dielectric - kr); 
                *p = std::exp(phase);
                p++;
            }
        );
    }

}