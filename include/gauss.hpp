#pragma once
#include "kernel_loop.hpp"
namespace kernels
{
    template <class T, size_t N, bool is_anisotropic = false> inline void gauss(T* p, const std::array<size_t, N>& shape, T sigma) 
    {
        if(0 >= sigma) // TODO : palse
        {
            std::fill(p, p + std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>()), T(0));
            return;
        }
        std::array<T, N> sigma_scala = step_from_shape<T, N, is_anisotropic>(shape);
        
        const T sqrt_2pi = std::sqrt(T(M_PI * 2));
        const T norm = std::accumulate(sigma_scala.begin(), sigma_scala.end(),
            T(1.0) / std::pow(sigma * sqrt_2pi, N),
            [](T a, T b){return a * b;}
        );
        const T sigma_2 = 2 * sigma * sigma;
        
        center_zero_loop_distance_2<T, N>(shape, sigma_scala, 
            [&](const std::array<T, N>& pos, T r){
                *p = norm * std::exp(-r / sigma_2); 
                p++;
            }
        );
    }
}