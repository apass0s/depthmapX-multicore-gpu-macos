// Copyright (C) 2026 Tasos Varoudis

#include "catch.hpp"
#include "genlib/pafmath.h"
#include "salalib/axialmap.h"
#include "salalib/axialmodules/axialintegration.h"
#include "salalib/mgraph.h"
#include "salalib/vgamodules/vgametric.h"
#include "salalib/vgamodules/vgametricdepth.h"

TEST_CASE("Axial Penn outputs use their distinct normalisations", "[penn]") {
    ShapeGraph map("Penn test");
    map.initialiseAttributesAxial();
    map.makeLineShape(Line(Point2f(0.0, 0.0), Point2f(1.0, 0.0)));
    map.makeLineShape(Line(Point2f(1.0, 0.0), Point2f(1.0, 1.0)));
    map.makeLineShape(Line(Point2f(1.0, 1.0), Point2f(2.0, 1.0)));
    map.makeLineShape(Line(Point2f(2.0, 1.0), Point2f(2.0, 2.0)));
    map.makeLineShape(Line(Point2f(2.0, 2.0), Point2f(3.0, 2.0)));
    map.makeConnections();

    REQUIRE(AxialIntegration({-1.0, 1.0, 2.0}, -1, false, true, false).run(nullptr, map, false));

    const AttributeTable &attributes = map.getAttributeTable();
    const AttributeRow &endpoint = attributes.getRow(AttributeKey(0));
    const double endpointRa = 1.0;
    REQUIRE(endpoint.getValue("Integration [HH]") == Approx(dvalue(5.0) / endpointRa));
    REQUIRE(endpoint.getValue("Integration [P-value]") == Approx(pvalue(5.0) / endpointRa));
    REQUIRE(endpoint.getValue("Integration [P-value]") != Approx(endpoint.getValue("Integration [HH]")));
    REQUIRE(endpoint.getValue("RA [Penn]") == Approx(0.0));
    REQUIRE(endpoint.getValue("RA [Penn] R1") == -1.0f);
    REQUIRE(endpoint.getValue("RA [Penn] R2") == Approx(0.0));

    const AttributeRow &middle = attributes.getRow(AttributeKey(2));
    REQUIRE(middle.getValue("RA [Penn]") == Approx(1.0 / 3.0));
    REQUIRE(middle.getValue("RA [Penn] R1") == -1.0f);
    REQUIRE(middle.getValue("RA [Penn] R2") == Approx(1.0 / 3.0));
}

TEST_CASE("Current VGA analyses expose legacy Penn detour measures", "[penn]") {
    const double spacing = 1.0;
    const Point2f offset(0.0, 0.0);
    const QtRegion region(Point2f(0.0, 0.0), Point2f(2.0, 2.0));
    const std::vector<SpacePixelFile> drawingFiles;
    PointMap map(region, drawingFiles, "Penn VGA test");
    REQUIRE(map.setGrid(spacing, offset));
    REQUIRE(map.makePoints(Point2f(1.0, 1.0), 0));
    REQUIRE(map.sparkGraph2(nullptr, false, -1.0));

    const PixelRef root(1, 1);
    map.setCurSel({int(root)});
    REQUIRE(VGAMetricDepth().run(nullptr, map, false));

    const AttributeTable &depthAttributes = map.getAttributeTable();
    for (auto iter = depthAttributes.begin(); iter != depthAttributes.end(); ++iter) {
        const AttributeRow &row = iter->getRow();
        REQUIRE(row.getValue("Metric Step Penn Distance") ==
                Approx(row.getValue("Metric Step Shortest-Path Length") -
                       row.getValue("Metric Straight-Line Distance")));
    }

    REQUIRE(VGAMetric(-1.0, false).run(nullptr, map, false));
    const AttributeTable &metricAttributes = map.getAttributeTable();
    for (auto iter = metricAttributes.begin(); iter != metricAttributes.end(); ++iter) {
        const AttributeRow &row = iter->getRow();
        REQUIRE(row.getValue("Metric Mean Penn Distance") ==
                Approx(row.getValue("Metric Mean Shortest-Path Distance") -
                       row.getValue("Metric Mean Straight-Line Distance")));
    }
}
