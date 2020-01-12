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
    int batch_size;
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
    // gid = [batch, depth, row * col]
    int i = gid[2] / info.out_size[1];
    int j = gid[2] % info.out_size[1];
    int ri = i * info.stride;
    int rj = j * info.stride;
    int2 max_pos = {ri, rj};
    float maxv = input[gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                     gid[1] * info.in_size[1] * info.in_size[2] +
                     ri * info.in_size[2] +
                     rj];
    
    for (int x = 0; x < info.core_size[0]; x++) {
        for (int y = 0; y < info.core_size[1]; y++) {
            int rx = ri + x, ry = rj + y;
            float curv = input[gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                             gid[1] * info.in_size[1] * info.in_size[2] +
                             rx * info.in_size[2] +
                             ry];
            
            if (maxv < curv) {
                maxv = curv;
                max_pos = {rx, ry};
            }
        }
    }
    
    int index =
        gid[0] * info.in_size[0] * info.out_size[0] * info.out_size[1] +
        gid[1] * info.out_size[0] * info.out_size[1] +
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
