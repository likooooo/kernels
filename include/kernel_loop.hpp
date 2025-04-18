#pragma once
#include <array>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <type_traist_notebook/type_traist.hpp>

namespace kernels
{
    template<class TPos, size_t N, class PixelFunc, size_t Dim = 0>
    constexpr inline void __kernel_loop_impl(const std::array<size_t, N>& shape, const std::array<TPos, N>& center, PixelFunc&& func, std::array<size_t, N>& indices) 
    {
        if constexpr (Dim < N) 
        {
            for (indices[Dim] = 0; indices[Dim] < shape[Dim]; ++indices[Dim]) 
            {
                __kernel_loop_impl<TPos, N, PixelFunc, Dim + 1>(shape, center, std::forward<PixelFunc>(func), indices);
            }
        } 
        else 
        {
            func(center, indices);
        }
    }

    template<class T, size_t N, class PixelFunc>
    constexpr inline void kernel_loop(const std::array<size_t, N>& shape, PixelFunc&& func) 
    {
        std::array<T, N> center{};
        // std::transform(shape.begin(), shape.end(), center.begin(), [](size_t i){return T(i) / 2;});
        for(size_t i = 0; i < N; i++) center.at(i) = shape.at(i) / 2;
        std::array<size_t, N> indices{};
        __kernel_loop_impl<T>(shape, center, std::forward<PixelFunc>(func), indices);
    }
    template<class T,size_t N> constexpr inline std::array<T, N> default_step(T step = T(1))
    {
        std::array<T, N> isotropic_step{};
        for(size_t i = 0; i < N; i++) isotropic_step[i] = step;
        return isotropic_step;
    }
    template<class T, size_t N, bool is_anisotropic = true> constexpr inline std::array<T, N> step_from_shape(const std::array<size_t, N>& shape, T isotropic_step = T(1)/* useless if anisotropic is true */)
    {
        if constexpr(is_anisotropic){
            return default_step<T, N>(isotropic_step);
        }
        else{
            std::array<T, N> sigma_scala{};
            for(size_t i = 0; i < N; i++) sigma_scala.at(i) = T(shape.back())/ T(shape.at(i));
            // std::transform(shape.begin(), shape.end(), sigma_scala.begin(), [&](size_t n){
            //      return  T(shape.back())/ T(n);
            // });
            return sigma_scala;
        }
    }
    template<class T, size_t N, class PixelFunc> constexpr inline void center_zero_loop_square_r(const std::array<size_t, N>& shape, const std::array<T, N>& step, PixelFunc&& func)
    {
        kernel_loop<T, N>(shape, [&](const std::array<T, N>& center, const std::array<size_t, N>& indices) {
            std::array<T, N> index_center_zero{};
            T sum_sq = 0;
            for (size_t i = 0; i < N; ++i) 
            {
                index_center_zero[i] = (static_cast<T>(indices[i]) - center[i]) * step[i];
                sum_sq += index_center_zero[i] * index_center_zero[i];
            }
            func(index_center_zero, sum_sq);
        });
    }
    template<class T, size_t N, class PixelFunc> constexpr inline void corner_zero_loop_square_r(const std::array<size_t, N>& shape, const std::array<T, N>& step, PixelFunc&& func)
    {
        kernel_loop<T, N>(shape, [&](const std::array<T, N>& center, const std::array<size_t, N>& indices) {
            std::array<T, N> pos{};
            T sum_sq = 0;
            
            for (size_t i = 0; i < N; ++i) 
            {
                size_t n = indices[i] - shape[i] * int(indices[i] >= shape[i]/2);
                pos[i] = static_cast<T>(n) * step[i];
                sum_sq += pos[i] * pos[i];
            }
            func(pos, sum_sq);
        });
    }
}