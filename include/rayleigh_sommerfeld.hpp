#pragma once
#include "kernel_loop.hpp"
namespace kernels
{
    //== from virtual lab
    template<class T> inline void rayleigh_sommerfeld_kernel(complex_t<T>* p, const vec2<size_t>& shape, const vec2<T>& step, T lambda,T dz)
    {
        complex_t<T> ik = 2_PI_I / lambda;
        T dz_2 = dz * dz;
        kernels::center_zero_loop_square_r<T, 2>(shape, step, 
            [&](const vec2<T> fyx, T r_2){
                T r = std::sqrt(r_2 + dz_2); 
                complex_t<T> n = 0;
                if(0 != r){
                    n = dz * std::exp(ik * r) *(T(1) - ik * r);
                    n /= (T(2_PI) * std::pow<T>(r, 3));
                }
                *p = n;
                p++;
            }
        );
    }
}