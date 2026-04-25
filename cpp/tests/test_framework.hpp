// Minimalny self-contained test framework. Zaden zewnetrzny gtest/catch.
// Test = funkcja void(); rejestrujesz przez TEST(name, code).
// Asercja: ASSERT_NEAR(a, b, tolerance), ASSERT_EQ(a, b), ASSERT_TRUE(cond).
#pragma once

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dtest {

struct TestCase {
    const char* name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct AutoRegister {
    AutoRegister(const char* name, void (*fn)()) {
        registry().push_back({name, fn});
    }
};

inline int run_all() {
    int passed = 0, failed = 0;
    std::vector<std::string> failures;
    for (const auto& tc : registry()) {
        std::cout << "[ RUN      ] " << tc.name << "\n";
        try {
            tc.fn();
            std::cout << "[       OK ] " << tc.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "[  FAILED  ] " << tc.name << ": " << e.what() << "\n";
            failures.push_back(std::string(tc.name) + ": " + e.what());
            ++failed;
        }
    }
    std::cout << "\n=== " << passed << " passed, " << failed << " failed ===\n";
    if (failed > 0) {
        std::cout << "\nFailures:\n";
        for (const auto& f : failures) std::cout << "  " << f << "\n";
    }
    return failed == 0 ? 0 : 1;
}

}  // namespace dtest

#define TEST(NAME)                                                     \
    static void NAME();                                                \
    static dtest::AutoRegister _reg_##NAME(#NAME, NAME);               \
    static void NAME()

#define ASSERT_TRUE(COND)                                              \
    do {                                                               \
        if (!(COND)) {                                                 \
            std::ostringstream _oss;                                   \
            _oss << "ASSERT_TRUE failed: " #COND " at " << __FILE__    \
                 << ":" << __LINE__;                                   \
            throw std::runtime_error(_oss.str());                      \
        }                                                              \
    } while (0)

#define ASSERT_FALSE(COND) ASSERT_TRUE(!(COND))

#define ASSERT_EQ(A, B)                                                \
    do {                                                               \
        auto _a = (A);                                                 \
        auto _b = (B);                                                 \
        if (!(_a == _b)) {                                             \
            std::ostringstream _oss;                                   \
            _oss << "ASSERT_EQ failed: " << #A " (" << _a              \
                 << ") != " #B " (" << _b << ") at " << __FILE__       \
                 << ":" << __LINE__;                                   \
            throw std::runtime_error(_oss.str());                      \
        }                                                              \
    } while (0)

#define ASSERT_NEAR(A, B, TOL)                                         \
    do {                                                               \
        double _a = static_cast<double>(A);                            \
        double _b = static_cast<double>(B);                            \
        double _t = static_cast<double>(TOL);                          \
        if (std::fabs(_a - _b) > _t) {                                 \
            std::ostringstream _oss;                                   \
            _oss << "ASSERT_NEAR failed: " << #A " (" << _a            \
                 << ") not within " << _t << " of " #B " (" << _b      \
                 << "), diff=" << std::fabs(_a - _b)                   \
                 << " at " << __FILE__ << ":" << __LINE__;             \
            throw std::runtime_error(_oss.str());                      \
        }                                                              \
    } while (0)
