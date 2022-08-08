#include <iostream>
#include <vector>
#include <cassert>

#include "../src/cnum.h"

void test_tensor_props() {
    cn::DTensor t1{ {1.0, 2.0, 3.0, 4.0}, {4} };

    assert(t1.shape.at(0) == 4);
}

void test_tensor_add() {

    cn::DTensor t1{ {1.0, 2.0, 3.0, 4.0}, {4} };

    cn::DTensor t2{ {1.0, 2.0, 3.0, 4.0}, {4} };

    double scalar = 1.0;

    assert((t1 + t2).data == (std::vector<double>{2.0, 4.0, 6.0, 8.0}));

    assert((t1 + scalar).data == (std::vector<double>{2.0, 3.0, 4.0, 5.0}));
}

void test_tensor_sub() {
    cn::DTensor t1{ {8.0, 9.0, 4.0}, {3} };

    cn::DTensor t2{ {3.0, 3.0, 7.0}, {3} };

    double scalar = 3.0;

    assert((t1 - t2).data == (std::vector<double>{5.0, 6.0, -3.0}));

    assert((t1 - scalar).data == (std::vector<double>{5.0, 6.0, 1.0}));
}

void test_tensor_mul() {
    cn::DTensor t1{ {6.0, 2.0}, {2} };

    cn::DTensor t2{ {8.0, 7.0}, {2} };

    double scalar = 2.0;

    assert((t1 * t2).data == (std::vector<double>{48.0, 14.0}));
    assert((t1 * scalar).data == (std::vector<double>{12.0, 4.0}));
}

void test_tensor_div() {
    cn::DTensor t1{ {2.0, 9.0, 1.0, 4.0}, {4} };

    cn::DTensor t2{ {4.0, 2.0, 2.0, 2.0}, {4} };

    double scalar = 4.0;

    assert((t1 / t2).data == (std::vector<double>{0.5, 4.5, 0.5, 2.0}));
    assert((t1 / scalar).data == (std::vector<double>{0.5, 2.25, 0.25, 1.0}));
}

int main() {
    test_tensor_props();
    test_tensor_add();
    test_tensor_sub();
    test_tensor_mul();
    test_tensor_div();
    return 0;
}