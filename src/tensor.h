#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <iterator>

namespace cn {
    // Represents a tensor.
    template <typename T>
    class Tensor {
    public:
        // Represents the flattened data of the tensor.
        std::vector<T> data{};

        // Represents the shape of the tensor, or how long it extends in each dimension.
        std::vector<size_t> shape{};

        // Constructs a tensor with all-zeroes from it's shape.
        Tensor(const std::vector<size_t>& shape);

        // // Constructs a tensor with all-zeroes from it's shape.
        // Tensor(const std::initializer_list<size_t>& shape);

        // Constructs a tensor from it's data and shape.
        Tensor(const std::vector<T>& data, const std::vector<size_t>& shape);

        // Constructs a tensor from it's data and shape.
        Tensor(const std::initializer_list<T>& data, const std::initializer_list<size_t>& shape);

        // Returns the component-wise sum of two tensors.
        Tensor<T> operator+(const Tensor<T>& other) const;

        // Returns the component-wise sum of a tensor and a scalar.
        Tensor<T> operator+(T other) const;

        // Returns the component-wise difference of two tensors.
        Tensor<T> operator-(const Tensor<T>& other) const;

        // Returns the component-wise difference of a tensor and a scalar.
        Tensor<T> operator-(T other) const;

        // Returns the component-wise product of two tensors.
        Tensor<T> operator*(const Tensor<T>& other) const;

        // Returns the component-wise product of a tensor and a scalar.
        Tensor<T> operator*(T other) const;

        // Returns the component-wise quotient of two tensors.
        Tensor<T> operator/(const Tensor<T>& other) const;

        // Returns the component-wise quotient of a tensor and a scalar.
        Tensor<T> operator/(T other) const;
    };

    typedef Tensor<unsigned short> USTensor;
    typedef Tensor<short> STensor;
    typedef Tensor<int> ITensor;
    typedef Tensor<unsigned int> UITensor;
    typedef Tensor<long> LTensor;
    typedef Tensor<unsigned long> ULTensor;
    typedef Tensor<long long> LLTensor;
    typedef Tensor<unsigned long long> ULLTensor;
    typedef Tensor<float> FTensor;
    typedef Tensor<double> DTensor;

    template <typename T>
    Tensor<T>::Tensor(const std::vector<size_t>& shape)
        : shape(shape) {
    }

    template <typename T>
    Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
        : data(data), shape(shape) {
    }

    template <typename T>
    Tensor<T>::Tensor(const std::initializer_list<T>& data, const std::initializer_list<size_t>& shape)
        : data(data), shape(shape) {
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        assert(data.size() == other.data.size());

        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            other.data.begin(),
            std::back_inserter(result.data),
            [](T n, T m) { return n + m; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator+(T other) const {
        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            std::back_inserter(result.data),
            [other](T n) { return n + other; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const {
        assert(data.size() == other.data.size());

        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            other.data.begin(),
            std::back_inserter(result.data),
            [](T n, T m) { return n - m; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator-(T other) const {
        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            std::back_inserter(result.data),
            [other](T n) { return n - other; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
        assert(data.size() == other.data.size());

        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            other.data.begin(),
            std::back_inserter(result.data),
            [](T n, T m) { return n * m; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator*(T other) const {
        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            std::back_inserter(result.data),
            [other](T n) { return n * other; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const {
        assert(data.size() == other.data.size());

        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            other.data.begin(),
            std::back_inserter(result.data),
            [](T n, T m) { return n / m; });

        return result;
    }

    template <typename T>
    Tensor<T> Tensor<T>::operator/(T other) const {
        Tensor<T> result{ shape };

        std::transform(data.begin(), data.end(),
            std::back_inserter(result.data),
            [other](T n) { return n / other; });

        return result;
    }
    
} // namespace cn

#endif // TENSOR_H
