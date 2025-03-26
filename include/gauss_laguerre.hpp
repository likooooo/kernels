#pragma once
#include <cmath>
#include <memory>
#include <iostream>
#include "kernel_loop.hpp"

namespace kernels{
    template<class TReal> inline TReal evaluate_laguerre(
        TReal x, int laguerre_order, int associated_order) 
    {
        if (laguerre_order == 0) {
            return 1.0;
        }
        if (laguerre_order == 1) {
            return 1.0 - x + laguerre_order;
        }

        TReal L_prev2 = 1.0; // L_{0}^{m}(x)
        TReal L_prev1 = 1.0 - x + associated_order; // L_{1}^{m}(x)
        TReal L_current = 0.0;

        for (int p = 2; p <= laguerre_order; ++p) {
            L_current = ((2 * p + associated_order - 1 - x) * L_prev1 -
                        (p + associated_order - 1) * L_prev2) / p;
            L_prev2 = L_prev1;
            L_prev1 = L_current;
        }
        return L_current;
    }

    template<class T, size_t N> inline complex_t<T> associated_theta(
        int associated_order, const std::array<real_t<T>, N>& pos, real_t<T> sigma)
    {
        if(associated_order == 0) {
            return complex_t<T>(1);
        }
        else if(associated_order == 1){
            return complex_t<T>(pos[1], pos[0]) / sigma;
        }
        else if(associated_order == 2){
            auto theta = complex_t<T>(pos[1], pos[0]) / sigma;
            real_t<T> cos = theta.real();
            real_t<T> sin = theta.imag();
            return complex_t<T>(
                std::pow(cos,2.0) - std::pow(sin,2.0),
                cos * sin * 2
            );
        }
        else{
            real_t<T> theta = std::atan2(pos[0], pos[1]);
            return std::exp(complex_t<T>(1_I) * (3 * theta)) * std::hypot(pos[1], pos[0])/sigma;
        }
        // printf("unimplement associated_order=%d\n", associated_order);
        return 1;
    }


    template<class T, size_t N, bool is_anisotropic = false> void gauss_laguerre(
        T* p, const std::array<size_t, N>& shape, real_t<T> sigma, int laguerre_order, int associated_order)
    {
        using real = real_t<T>;
        const std::array<real, N> steps = step_from_shape<real, N, is_anisotropic>(shape);
        const real sigma_2 = sigma * sigma;
        const real norm = 1.0 / (M_PI * sigma_2);
        center_zero_loop_square_r<real, N>(shape, steps, 
            [&](const std::array<real, N>& pos, real r){
                real t = r / sigma_2;
                complex_t<T> theta = norm * associated_theta<T, 2>(associated_order, pos, 1);
                // if(r <= sigma)
                {
                    if constexpr(is_complex_v<T>){
                        *p = theta * std::exp(-t) * evaluate_laguerre(2 * t, laguerre_order, associated_order);
                    }
                    else{
                        *p = std::real(theta * std::exp(-t) * evaluate_laguerre(2 * t, laguerre_order, associated_order));
                    }
                }
                // else{
                //     *p = 0;
                // }
                p++;
            }
        );
    }
}