// depthmapX GPU module - attribute rows of a shape map, indexed by shape index

#pragma once

#include "salalib/shapemap.h"

#include <vector>

namespace dmxaccel {

// rows[i] == &map.getAttributeRowFromShapeIndex(i), built in one pass. The upstream
// accessor walks the shape std::map from its start on every call (O(i)), which makes
// loops over all shapes quadratic (found by the rapido build, changelog item 6).
inline std::vector<AttributeRow *> rowsByShapeIndex(ShapeMap &map) {
    std::vector<AttributeRow *> rows;
    rows.reserve(map.getShapeCount());
    AttributeTable &table = map.getAttributeTable();
    for (auto &shape : map.getAllShapes()) {
        rows.push_back(&table.getRow(AttributeKey(shape.first)));
    }
    return rows;
}

} // namespace dmxaccel
