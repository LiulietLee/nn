//
//  inception_m.metal
//  nn
//
//  Created by Liuliet.Lee on 20/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sum_inter_delta(device const float *data,
                            device const int &core_size,
                            device const int &len,
                            device float *res,
                            uint index [[ thread_position_in_grid ]])
{
    for (int i = 0; i < core_size; i++) {
        res[index] += data[i * len + index];
    }
}
