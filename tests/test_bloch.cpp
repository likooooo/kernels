#include <bloch.hpp>
#include <vector>
#include <py_helper.hpp>
#include <functional>

template<class T, bool is_anisotropic = false>
void test(size_t xsize, size_t ysize, T freq, T crao, T azimuth)
{
    if(is_anisotropic) {
        size_t n(0.9 * ysize);
        ysize = n == ysize? ysize - 1 : n;
    }
    printf("* bloch_phase<%s, %s> test with shape=(%zu, %zu) crao=%f, azimuth=%f, freq=%f\n", 
        TypeReflection<T>().c_str(), 
        is_anisotropic ? "anisotropic" : "isotropic",
        ysize, xsize, crao, azimuth, freq
    );

    // compare bloch phase with shift-kernel
    std::vector<complex_t<T>> vec(xsize * ysize);
    kernels::bloch_phase<T, 2>(vec.data(), {ysize, xsize}, freq, crao, azimuth);
    std::vector<T> real_part(xsize * ysize);
    std::transform(vec.begin(), vec.end(), real_part.begin(), [](auto c){return c.real();});

    kernels::phase_modulate<T, 2>(vec.data(), {ysize, xsize}, {std::sin(crao) * std::sin(azimuth)*freq, std::sin(crao) * std::cos(azimuth)*freq});
    std::vector<T> imag_part(xsize * ysize);
    std::transform(vec.begin(), vec.end(), imag_part.begin(), [](auto c){return c.real();});
    imshow(real_part, {(int)xsize, (int)ysize});
    imshow(imag_part, {(int)xsize, (int)ysize});
    // imshow(real_part - imag_part, {(int)xsize, (int)ysize});
}

int main()
{
    py_loader::init();
    py_plot::get_default_visualizer_dir() = "/usr/local/bin";
    // TODO : test case for 1D & 3D
    for(auto freq :{1, 2})
    for(auto crao : {0.5, 0.25})
    for(auto azimuth : {0.0, 0.5, 0.25})
    for(size_t x : {256})
    {
        constexpr bool enable_anisotropic = true;
        test<double, !enable_anisotropic>(x, x, freq, crao * M_PI, azimuth * M_PI);
        
        std::vector<complex_t<double>> vec(x * x);
        kernels::free_propagation<double, 2>(vec.data(), {x, x}, kernels::default_step<double, 2>(0.01), freq, crao);
        std::vector<double> real_part(x * x);
        std::transform(vec.begin(), vec.end(), real_part.begin(), [](auto c){return c.real();});
        imshow(real_part, {(int)x, (int)x});
    }
}