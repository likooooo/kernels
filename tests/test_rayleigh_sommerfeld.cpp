#include <rayleigh_sommerfeld.hpp>
#include <py_helper.hpp>
int main()
{
    py_engine::init();
    using rT =float;
    using cT = std::complex<rT>;

    size_t xsize = 256;
    size_t ysize = xsize;
    rT physical_x = 50;
    rT physical_y = physical_x;
    rT lambda = 13.5;
    std::vector<cT> vec(xsize * ysize);
    // for(rT dz :{5, 10})
    // {
    //     kernels::rayleigh_sommerfeld_kernel(vec.data(), {ysize, xsize}, {physical_y / ysize, physical_x / xsize},  lambda, dz);
    //     const auto&[real, imag] = decompose_from<cT, rT, rT>(vec);

    //     printf("* test result dz=%f\n    real-part\n", dz);
    //     imshow(real, {xsize, ysize});
    //     printf("    imag-part\n");
    //     imshow(imag, {xsize, ysize});
    //     printf("    norm\n");
    //     imshow(norm(vec), {xsize, ysize});
    // }
    auto display = py_plot::create_callback_simulation_fram_done(py::object(overload_click));
    for(rT dz =0; dz < 2 * lambda; dz += (0.1 * lambda)){
        kernels::rayleigh_sommerfeld_kernel(vec.data(), {ysize, xsize}, {physical_y / ysize, physical_x / xsize},  lambda, dz);
        auto[real, imag] = decompose_from<cT, rT, rT>(vec);
        // auto real = norm(vec);
        rT max = *std::max_element(real.begin(), real.end());
        if(0 != max) real /= max;
        display(create_ndarray_from_vector(real, convert_to<std::vector<int>>(vec2<size_t>{xsize, ysize})));
        sleep(1);
    }
}