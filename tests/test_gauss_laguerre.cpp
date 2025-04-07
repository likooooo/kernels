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
BOOST_PYTHON_MODULE(lib_test_gauss_laguerre) 
{
    py_engine::init();
    py_engine::init_exception_for_pycall();
    boost::python::def("gauss_laguerre",  test<double, false>);
    boost::python::def("gauss_laguerreV1",  test<double, true>);
}
