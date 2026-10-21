// Copyright (C) 2026 Tasos Varoudis

#include "catch.hpp"
#include "genlib/pafmath.h"
#include "mgraph440/pafmath.h"

#include <utility>
#include <vector>

TEST_CASE("Current and legacy Penn palm-tree helpers agree", "[penn]") {
    const std::vector<std::pair<double, double>> inputs{{-1.0, 2.0}, {0.0, 0.0}, {1.0, 0.0},
                                                        {5.0, -1.0}, {5.0, 1.0}, {5.0, 4.0},
                                                        {5.0, 5.0},  {5.0, 6.0}};
    for (const auto &input : inputs) {
        REQUIRE(palmtree(input.first, input.second) == mgraph440::palmtree(input.first, input.second));
    }

    REQUIRE(palmtree(5.0, 4.0) == 10.0);
    REQUIRE(palmtree(5.0, 5.0) == 10.0);
    REQUIRE(palmtree(5.0, 6.0) == 10.0);
    REQUIRE(palmtree(5.0, -1.0) == 0.0);
    REQUIRE(palmtree(-1.0, 2.0) == 0.0);
}
