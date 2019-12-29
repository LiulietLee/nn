//
//  maxpooling_m.metal
//  nn
//
//  Created by Liuliet.Lee on 29/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct pooling_layer_info {
    int2 core_size;
    int2 out_size;
    int3 in_size;
    int stride;
};

struct switch_mapper {
    int oi, oj, ix, iy, k;
};

kernel void maxpooling_forward(device const pooling_layer_info &info,
                               device const float *input,
                               device switch_mapper *switches,
                               device float *score,
                               uint3 gid [[ thread_position_in_grid ]])
{
    int ri = gid[0] * info.stride;
    int rj = gid[1] * info.stride;
    int2 max_pos = {ri, rj};
    int maxv = input[ri * info.in_size[1] * info.in_size[2] +
                     rj * info.in_size[2] +
                     gid[2]];
    
    for (int x = 0; x < info.core_size[0]; x++) {
        for (int y = 0; y < info.core_size[1]; y++) {
            int rx = ri + x, ry = rj + y;
            int curv = input[rx * info.in_size[1] * info.in_size[2] +
                             ry * info.in_size[2] +
                             gid[2]];
            if (maxv < curv) {
                maxv = curv;
                max_pos = {rx, ry};
            }
        }
    }
    
    int index =
        gid[0] * info.out_size[1] * info.in_size[2] +
        gid[1] * info.in_size[2] +
        gid[2];
    
    switches[index] = switch_mapper {
        (int)gid[0], (int)gid[1], max_pos[0], max_pos[1], (int)gid[2]
    };
    score[index] = maxv;
}

kernel void maxpooling_backward(device const pooling_layer_info &info,
                                device const switch_mapper *switches,
                                device const float *delta,
                                device float *da,
                                uint index [[ thread_position_in_grid ]])
{
    switch_mapper m = switches[index];
    da[m.ix * info.in_size[1] * info.in_size[2] +
       m.iy * info.in_size[2] +
       m.k]
    =
    delta[m.oi * info.out_size[1] * info.in_size[2] +
          m.oj * info.in_size[2] +
          m.k];
}
