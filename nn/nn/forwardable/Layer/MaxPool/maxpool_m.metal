//
//  maxpool_m.metal
//  nn
//
//  Created by Liuliet.Lee on 29/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "../../../utils/Metal/metal_utils.h"

struct pooling_layer_info {
    int2 core_size;
    int2 out_size;
    int3 in_size;
    int stride;
    int padding;
    int batch_size;
};

struct switch_mapper {
    int batch;
    int3 inpos, outpos;
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
    float maxv = -MAXFLOAT;
    if (in_bound(ri, rj, info.in_size[1], info.in_size[2])) {
        maxv = input[gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                     gid[1] * info.in_size[1] * info.in_size[2] +
                     ri * info.in_size[2] +
                     rj];
    }
    
    for (int x = 0; x < info.core_size[0]; x++) {
        for (int y = 0; y < info.core_size[1]; y++) {
            int rx = ri + x - info.padding, ry = rj + y - info.padding;
            if (!in_bound(rx, ry, info.in_size[1], info.in_size[2])) {
                continue;
            }
            
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
    
    int3 inpos = {(int)gid[1], max_pos[0], max_pos[1]};
    int3 outpos = {(int)gid[1], i, j};
    switches[index] = switch_mapper {
        (int)gid[0], inpos, outpos
    };
    score[index] = maxv;
}

kernel void maxpooling_backward(device const pooling_layer_info &info,
                                device const switch_mapper *switches,
                                device const float *delta,
                                device float *da,
                                uint3 gid [[ thread_position_in_grid ]])
{
    int batch = gid[0];
    int k = gid[1];
    int ri = gid[2] / info.in_size[2];
    int rj = gid[2] % info.in_size[2];
    
    int pos =
    batch * info.in_size[0] * info.in_size[1] * info.in_size[2] +
    k * info.in_size[1] * info.in_size[2] +
    ri * info.in_size[2] +
    rj;
    
    int offi = (ri + info.padding) / info.core_size[0];
    int offj = (rj + info.padding) / info.core_size[1];
    
    for (int i = offi - info.core_size[0] + info.padding + 1; i <= offi + info.padding; i++) {
        for (int j = offj - info.core_size[1] + info.padding + 1; j <= offj + info.padding; j++) {
            if (in_bound(i, j, info.out_size[0], info.out_size[1])) {
                int index =
                batch * info.in_size[0] * info.out_size[0] * info.out_size[1] +
                k * info.out_size[0] * info.out_size[1] +
                i * info.out_size[1] +
                j;

                switch_mapper m = switches[index];

                if (m.inpos[1] == ri && m.inpos[2] == rj) {
                    da[pos] += delta[index];
                }
            }
        }
    }
}
