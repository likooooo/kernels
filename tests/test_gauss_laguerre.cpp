#include <gauss_laguerre.hpp>
#include <vector>
#include <py_helper.hpp>
#include <functional>
#include <mekil/mkl_wrapper.hpp>

template<class T, bool is_anisotropic = false>
void test(size_t xsize, size_t ysize, T sigma, int order, int associate)
{
    if(is_anisotropic) {
        size_t n(0.9 * ysize);
        ysize = n == ysize? ysize - 1 : n;
    }
    printf("* laguerre<%s, %s> test with shape=(%zu, %zu) sigma=%f\n", 
        TypeReflection<T>().c_str(), 
        is_anisotropic ? "anisotropic" : "isotropic",
        ysize, xsize, (float)std::abs(sigma)
    );

    std::vector<T> vec(xsize * ysize);
    kernels::gauss_laguerre<T, 2, is_anisotropic>(vec.data(), {ysize, xsize}, sigma, order, associate);
    imshow(vec, {xsize, ysize});
}

template<class T, class FuncConvWithKernel> std::vector<std::vector<T>> gauss_laguerre_conv_linear(const std::array<size_t, 2> shape, T sigma, size_t max_associated_order, size_t max_laguerre_order, FuncConvWithKernel&& conv_image_with)
{
    std::vector<std::vector<T>> linears;
    linears.reserve(max_laguerre_order * max_associated_order);
    for(size_t laguerre_order = 0; laguerre_order < max_laguerre_order; laguerre_order++)
    for(size_t associated_order = 0; associated_order < max_associated_order; associated_order++)
    {
        std::vector<T> kernel(shape[0] * shape[1]);
        kernels::gauss_laguerre<T, 2, false>(kernel.data(), shape, sigma, laguerre_order, associated_order);
        conv_image_with(kernel);
        linears.push_back(std::move(kernel));
    }
}
template<class T> inline std::vector<std::vector<T>> gauss_laguerre_conv_quadratic(const std::vector<std::vector<T>>& linear_results)
{
    std::vector<std::vector<T>> quadratic_result;
    size_t N = linear_results.size();
    quadratic_result.reserve(N * (N + 1) / 2);
    // 对称正定矩阵，只用计算上三角部分
    for(int r = 0; r < N; r++)
    for(int c = r; c < N; c++){
        std::vector<T> prod;
        auto& a = linear_results.at(r);
        auto& b = linear_results.at(c);
        std::transform(a.begin(), a.end(), b.begin(), std::back_insert_iterator(prod), [](T l, T r){return l * std::conj(r);});
        quadratic_result.push_back(std::move(prod));
    }
    return quadratic_result;
}
template<class T> std::vector<T> quadratic_coefficients_diagonalize(const std::vector<T>& coef_of_quad, size_t eigen_value_count)
{
    std::vector<T> eigenval(eigen_value_count);
    LAPACKE_theev(LAPACK_COL_MAJOR, 'V', 'U', eigen_value_count, coef_of_quad.data(), eigen_value_count, eigenval.data());
    return eigenval;
}


int main()
{
    py_engine::init();
    // TODO : test case for 1D & 3D
    for(int associate  : {0, 1, 2})
    for(int order : {0, 1})
    // for(size_t x : {8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1280})
    for(size_t x : {256})
    {
        constexpr bool enable_anisotropic = true;
        // test<double, enable_anisotropic>(x, x, 0.1 * x, order);
        printf("associate_order=%d laguerre_order=%d\n", associate, order);
        test<double, !enable_anisotropic>(x, x, 0.1 * x, order, associate);
        // test<float, enable_anisotropic>(x, x, 0.1 * x, order);
        // test<float, !enable_anisotropic>(x, x, 0.1 * x, order);
    }
}