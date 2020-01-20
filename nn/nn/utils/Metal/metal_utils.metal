//
//  metal_utils.metal
//  nn
//
//  Created by Liuliet.Lee on 19/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "metal_utils.h"

bool in_bound(int x, int y, int row, int col) {
    return 0 <= x && x < row && 0 <= y && y < col;
}
