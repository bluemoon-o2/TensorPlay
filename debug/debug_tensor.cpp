#include <iostream>
#include <vector>
#include <algorithm>
#include "tensorplay/core/Tensor.h"

using namespace tensorplay;

int main() {
    try {
        std::cout << "Debug Tensor C++ Test" << std::endl;
        std::cout << "=====================" << std::endl;

        // Test 1: Create Zeros
        std::vector<int64_t> shape = {2, 3};
        Tensor t1 = Tensor::zeros(shape);
        std::cout << "Zeros {2, 3}:\n" << t1 << std::endl;

        // Test 2: Create Ones
        Tensor t2 = Tensor::ones({3, 2});
        std::cout << "Ones {3, 2}:\n" << t2 << std::endl;

        // Test 3: Create Arange
        Tensor t3 = Tensor::arange(0, 4);
        std::cout << "Arange 0..4:\n" << t3 << std::endl;

        // Test 3.5: Create Rand
        Tensor t_rand = Tensor::rand({2, 2});
        std::cout << "Rand {2, 2}:\n" << t_rand << std::endl;

        // Test 3.6: Create Full
        Tensor t_full = Tensor::full({2, 2}, 3.14);
        std::cout << "Full {2, 2} with 3.14:\n" << t_full << std::endl;

        // Test 4: Manual Data Access and Modification
        std::cout << "Modifying t1 data..." << std::endl;
        float* data = t1.data_ptr<float>();
        for(int i=0; i<t1.numel(); ++i) {
            data[i] = (float)i;
        }
        std::cout << "Modified t1 (0..N):\n" << t1 << std::endl;

        // Test 5: Reshape
        std::cout << "Reshaping t1..." << std::endl;
        Tensor t4 = t1.reshape({3, 2});
        std::cout << "Reshaped t1 to {3, 2}:\n" << t4 << std::endl;

        // Test 6: Copy (Same dtype, contiguous)
        std::cout << "Testing Copy (Contiguous)..." << std::endl;
        Tensor t_copy_dst = Tensor::zeros({2, 3});
        t_copy_dst.copy_(t1); // t1 is {2,3} float 0..5
        std::cout << "Copy of t1:\n" << t_copy_dst << std::endl;

        // Test 7: Copy (Different dtype)
        std::cout << "Testing Copy (Cast)..." << std::endl;
        Tensor t_int = Tensor::ones({2, 3}, DType::Int64);
        Tensor t_float_dst = Tensor::zeros({2, 3});
        t_float_dst.copy_(t_int);
        std::cout << "Copy of Int64 Ones to Float32:\n" << t_float_dst << std::endl;
        
        // Test 8: Copy (Non-contiguous)
        std::cout << "Testing Copy (Non-contiguous)..." << std::endl;
        // Create a non-contiguous slice
        Tensor t_base = Tensor::arange(0, 10).reshape({2, 5}); // {{0,1,2,3,4}, {5,6,7,8,9}}
        // Slice: dim 1, start 0, end 5, step 2 -> cols 0, 2, 4. Shape {2, 3}
        Tensor t_slice = t_base.slice(1, 0, 5, 2); 
        
        Tensor t_nc_dst = Tensor::zeros({2, 3});
        t_nc_dst.copy_(t_slice);
        std::cout << "Copy of non-contiguous slice:\n" << t_nc_dst << std::endl;

        // Test 9: Arithmetic Ops
        std::cout << "Testing Arithmetic Ops..." << std::endl;
        Tensor a = Tensor::ones({2, 2});
        Tensor b = Tensor::full({2, 2}, 2.0);
        
        Tensor c = a.add(b); // 1 + 2 = 3
        std::cout << "Add (1+2):\n" << c << std::endl;
        
        Tensor d = c.mul(b); // 3 * 2 = 6
        std::cout << "Mul (3*2):\n" << d << std::endl;
        
        Tensor e = d.sub(a, Scalar(2.0)); // 6 - 2*1 = 4
        std::cout << "Sub (6 - 2*1):\n" << e << std::endl;
        
        Tensor f = e.div(b); // 4 / 2 = 2
        std::cout << "Div (4/2):\n" << f << std::endl;

        // Test 9.5: Scalar Item and Operators
        std::cout << "Testing Scalar Item and Operators..." << std::endl;
        Tensor s1 = Tensor::full({1}, 10.0f);
        Scalar sc1 = s1.item();
        std::cout << "Item: " << sc1.toString() << std::endl;
        
        Scalar sc2(2.0f);
        Scalar sc3 = sc1 + sc2;
        std::cout << "Scalar Add: " << sc3.toString() << std::endl;
        
        if (sc3.to<float>() != 12.0f) {
             throw std::runtime_error("Scalar add failed");
        }
        
        // Test New Constructor
        std::cout << "Testing Scalar Constructor..." << std::endl;
        Tensor t_scalar_ctor({2, 2}, Scalar(3.14f));
        std::cout << "Tensor from Scalar(3.14f):\n" << t_scalar_ctor << std::endl;
        
        // Test 10: To (Type Conversion)
        std::cout << "Testing To..." << std::endl;
        Tensor t_float = Tensor::full({2, 2}, 3.5);
        Tensor t_long = t_float.to(DType::Int64);
        std::cout << "Converted to Int64:\n" << t_long << std::endl;

        // Test 10.5: Tensor Factory (C++ vector/initializer_list)
        std::cout << "Testing Tensor Factory..." << std::endl;
        Tensor t_vec = Tensor::tensor(std::vector<float>{1.1f, 2.2f, 3.3f});
        std::cout << "Tensor from vector {1.1, 2.2, 3.3}:\n" << t_vec << std::endl;
        
        Tensor t_init = Tensor::tensor({10, 20, 30}, DType::Int32); // Infer int32
        std::cout << "Tensor from init_list {10, 20, 30}:\n" << t_init << std::endl;
        if (t_init.dtype() != DType::Int32) {
             throw std::runtime_error("Tensor factory dtype inference failed for int");
        }

        // Test 10.6: Shape Class
        std::cout << "Testing Shape Class..." << std::endl;
        Tensor t_shape = Tensor::zeros({2, 3});
        std::cout << "Shape of {2, 3} tensor: " << t_shape.shape() << std::endl;
        if (t_shape.shape()[0] != 2 || t_shape.shape()[1] != 3) {
            throw std::runtime_error("Shape mismatch");
        }

        // Test 11: DataPtr with Lambda Deleter
        std::cout << "Testing DataPtr with Lambda Deleter..." << std::endl;
        bool deleter_called = false;
        {
            float* raw_data = new float[10];
            // Use lambda deleter capturing local variable
            DataPtr ptr(raw_data, [&](void* d) {
                delete[] static_cast<float*>(d);
                deleter_called = true;
                std::cout << "Lambda deleter called!" << std::endl;
            }, Device(DeviceType::CPU));
            
            Storage s(std::move(ptr), 10 * sizeof(float));
            Tensor t(s, {10}, DType::Float32);
            // t goes out of scope, storage goes out of scope, deleter should be called
        }
        if (deleter_called) {
            std::cout << "DataPtr Lambda Deleter Test Passed" << std::endl;
        } else {
            std::cerr << "DataPtr Lambda Deleter Test FAILED" << std::endl;
            return 1;
        }
        
        std::cout << "All tests passed!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
