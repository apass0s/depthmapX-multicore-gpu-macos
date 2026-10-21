// Copyright (C) 2026 Tasos Varoudis

#include "catch.hpp"
#include "mgraph440/attr.h"

TEST_CASE("Legacy Penn distances preserve unavailable values and clamp detours", "[penn]") {
    mgraph440::AttrHeader header;
    mgraph440::AttrVal attributes[mgraph440::NUM_PHYSICAL_ATTRIBUTES];
    header.reset(attributes);

    attributes[mgraph440::AttrHeader::METRIC_GRAPH_SIZE].intval = 5;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::MEAN_PENN_DIST, attributes) == -1.0);

    attributes[mgraph440::AttrHeader::TOTAL_METRIC_DEPTH].floatval = 10.0f;
    attributes[mgraph440::AttrHeader::TOTAL_EUCLID_DIST].floatval = 7.0f;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::MEAN_PENN_DIST, attributes) == Approx(0.6));

    attributes[mgraph440::AttrHeader::TOTAL_EUCLID_DIST].floatval = 11.0f;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::MEAN_PENN_DIST, attributes) == 0.0);

    attributes[mgraph440::AttrHeader::METRIC_POINT_DEPTH].floatval = 5.0f;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::POINT_PENN_DIST, attributes) == -1.0);

    attributes[mgraph440::AttrHeader::POINT_EUCLID_DIST].floatval = 3.0f;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::POINT_PENN_DIST, attributes) == 2.0);

    attributes[mgraph440::AttrHeader::POINT_EUCLID_DIST].floatval = 6.0f;
    REQUIRE(header.getAttr(mgraph440::AttrHeader::POINT_PENN_DIST, attributes) == 0.0);
}
