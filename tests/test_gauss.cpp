#include <gauss.hpp>
#include <vector>
#include <py_helper.hpp>
#include <functional>
template<class T> std::pair<std::string, bool> max_value_check(const std::vector<T> vec, size_t xsize, size_t ysize)
{
    const T* p = vec.data();
    T max_value = -1e6;
    size_t posx = 0, posy = 0;
    for(size_t y = 0; y < ysize; y++)
    {
        for(size_t x = 0; x < xsize; x++, p++)
        {
            if(max_value < *p)
            {
                max_value = *p;
                posx = x;
                posy = y;
            }
        }
    }
    return {"is max value in center", posx == xsize/2 && posy == ysize/2};
}

template<class T> std::pair<std::string, bool> integral_check(const std::vector<T> vec, size_t xsize, size_t ysize)
{
    T sum = std::accumulate(vec.begin(), vec.end(), T(0));
    T eps = std::abs(sum - 1);
    bool flag = eps < 9e-2;
    if(!flag){
        printf("    integral of gauss error=%e\n", eps);
    }
    return std::pair<std::string, bool>{"integral of gauss should be 1", flag};
}

template<class T, bool is_anisotropic = false>
void test(size_t xsize, size_t ysize, T sigma)
{
    if(is_anisotropic) {
        size_t n(0.9 * ysize);
        ysize = n == ysize? ysize - 1 : n;
    }
    printf("* gauss<%s, %s> with shape=(%zu, %zu) sigma=%f\n", 
        TypeReflection<T>().c_str(), 
        is_anisotropic ? "anisotropic" : "isotropic",
        ysize, xsize, (float)std::abs(sigma)
    );
    std::vector<std::function<std::pair<std::string, bool>(const std::vector<T> vec, size_t xsize, size_t ysize)>> check_list{
        max_value_check<T>, integral_check<T>
    };

    std::vector<T> vec(xsize * ysize);
    kernels::gauss<T, 2, is_anisotropic>(vec.data(), {ysize, xsize}, sigma);
    
    for(const auto& func : check_list){
        auto [str, flag] = func(vec, xsize, ysize);
        if(flag){
            printf("    %s check success\n", str.c_str());
        } else{
            printf("    \033[31m%s check failed\033[0m\n", str.c_str());
            imshow(vec, {xsize, ysize});
        }
    }
    printf("\n");
}
int main()
{
    py_engine::init();
    // TODO : test case for 1D & 3D
    for(auto ratio : {0.1, 0.12, 0.15, 0.2})
    for(size_t x : {8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1280})
    {
        constexpr bool enable_anisotropic = true;
        test<double, enable_anisotropic>(x, x, double(ratio) * x);
        test<double, !enable_anisotropic>(x, x, double(ratio) * x);
        test<float, enable_anisotropic>(x, x, float(ratio) * x);
        test<float, !enable_anisotropic>(x, x, float(ratio) * x);
    }
}