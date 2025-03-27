#include <bloch.hpp>
#include <vector>
#include <py_helper.hpp>
#include <functional>
template<class T, bool is_anisotropic = false> void test_bloch(size_t xsize, size_t ysize, T freq, T crao, T azimuth)
{
    if(is_anisotropic) {
        size_t n(0.9 * ysize);
        ysize = n == ysize? ysize - 1 : n;
    }
    std::cout << "* bloch test with (shape, lambda, crao(PI), azimuth(PI))=" << std::make_tuple(vec2<size_t>{xsize, ysize}, 1.0/freq, crao/T(1_PI), azimuth/T(1_PI)) << std::endl;

    auto display = [&](const std::string& name,  std::vector<complex_t<T>> vec)
    {
        std::vector<T> real_part(xsize * ysize);
        std::vector<T> imag_part(xsize * ysize);
        std::transform(vec.begin(), vec.end(), real_part.begin(), [](auto c){return c.real();});
        std::transform(vec.begin(), vec.end(), imag_part.begin(), [](auto c){return c.imag();});
        std::cout << "    realpart of " << name << std::endl;
        imshow(real_part, {xsize, ysize});
        std::cout << "    imagpart of " << name << std::endl;
        imshow(imag_part, {xsize, ysize});
    };
    
    // compare bloch phase with shift-kernel
    std::vector<complex_t<T>> bloch(xsize * ysize);
    kernels::bloch_phase<T, 2>(bloch.data(), {ysize, xsize}, freq, {std::sin(crao) * std::sin(azimuth)*freq, std::sin(crao) * std::cos(azimuth)*freq});
    display("bloch phase", bloch);

    vec2<T> shift_pixel{std::sin(crao) * std::sin(azimuth)*freq, std::sin(crao) * std::cos(azimuth)*freq};
    std::cout << "    phase modulate with shift " << shift_pixel << "(pixel)" << std::endl;
    std::vector<complex_t<T>> shift(xsize * ysize);
    kernels::phase_modulate<T, 2>(shift.data(), {ysize, xsize}, shift_pixel);
    display("phase modulate", shift);
    auto error = abs(shift-bloch);
    std::cout << "    error image. max error = " << *std::max_element(error.begin(), error.end()) << std::endl;
    imshow(error, {xsize, ysize});
}

void auto_test_bloch()
{
    for(auto freq :{1, 2})
    for(auto crao : {0.5, 0.25})
    for(auto azimuth : {0.0, 0.5, 0.25})
    for(size_t x : {256})
    {
        constexpr bool enable_anisotropic = true;
        test_bloch<double, !enable_anisotropic>(x, x, freq, crao * M_PI, azimuth * M_PI);
        
        std::vector<complex_t<double>> vec(x * x);
        kernels::free_propagation<double, 2>(vec.data(), {x, x}, kernels::default_step<double, 2>(0.01), freq, crao);
        std::vector<double> real_part(x * x);
        std::transform(vec.begin(), vec.end(), real_part.begin(), [](auto c){return c.real();});
        imshow(real_part, {x, x});
    }
}

BOOST_PYTHON_MODULE(lib_test_bloch) 
{
    py_engine::init();
    py_engine::init_exception_for_pycall();
    boost::python::def("test_bloch", test_bloch<double, false>);
    boost::python::def("auto_test_bloch", auto_test_bloch);
}

int main()
{
    py_engine::init();
    auto_test_bloch();
}