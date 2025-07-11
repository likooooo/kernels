#pragma once
#include "kernel_loop.hpp"
namespace kernels
{
    //== https://en.wikipedia.org/wiki/Fresnel_diffraction
    template<class T> inline void fresnel_propagation_coef(complex_t<T>* p, const vec2<size_t>& shape, const vec2<T>& step, T lambda,T dz, T A = 1)
    {
        const complex_t<T> ik(0, T(2_PI) / lambda);
        kernels::center_zero_loop_square_r<T, 2>(shape, step, 
            [&](const vec2<T> yx, T rho_2){
                T r = dz + rho_2 / (T(2) * dz);
                *p = complex_t<T>(0, -A) * std::exp(ik * r);
                p++;
            }
        );
    }
    template<class T> inline vec2<T> fresnel_propagtion_operator(complex_t<T>* pOrigin, const vec2<size_t>& shape, const vec2<T>& step, T lambda,T dz)
    {
        const T k = T(2_PI) / lambda;
        const complex_t<T> ik(0, k);
        const T step_ratio = k / (T(2_PI) * dz);
        complex_t<T>* p = pOrigin;
        kernels::center_zero_loop_square_r<T, 2>(shape, step, 
            [&](const vec2<T> fyx, T rho_2){
                *p *= std::exp(ik * rho_2 / (T(2) * dz));
                p++;
            }
        );
        // TODO : fft(pOrigin)

        std::vector<complex_t<T>> coef(shape[0] * shape[1]);
        vec2<T> scalar_step = step / step_ratio;
        fresnel_propagation_coef(coef.data(), shape, scalar_step, lambda, dz, A);
        // p *= coef
        return scalar_step;
    }
    template<class T> inline vec2<T> fresnel_propagtion_kernel(complex_t<T>* p, const vec2<size_t>& shape, const vec2<T>& step, T lambda,T dz)
    {
        const T k = T(2_PI) / lambda;
        const complex_t<T> ik(0, k);
        const T step_ratio = k / (T(2_PI) * dz);
        complex_t<T>* p = pOrigin;
        kernels::center_zero_loop_square_r<T, 2>(shape, step, 
            [&](const vec2<T> fyx, T rho_2){
                *p *= std::exp(ik * rho_2 / (T(2) * dz));
                p++;
            }
        );
    }
}