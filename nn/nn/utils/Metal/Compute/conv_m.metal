//
//  conv_m.metal
//  nn
//
//  Created by Liuliet.Lee on 26/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct conv_layer_info {
    int3 core_size; // [width, height, depth]
    int3 in_size;   // [width of input, height of input, depth]
    int3 out_size;  // [row, col, count]
    int stride;
    int padding;
};

bool in_bound(int x, int y, int3 input_size) {
    return 0 <= x && x < input_size[0] && 0 <= y && y < input_size[1];
}

kernel void conv_forward(device const conv_layer_info &info,
                         device const float *input,
                         device const float *core,
                         device const float *bi,
                         device float *score,
                         uint2 gid [[ thread_position_in_grid ]])
{
    int i = gid[0] / info.out_size[1];
    int j = gid[0] % info.out_size[1];
    
    score[gid[0] + gid[1]] = 0.0;
    for (int x = 0; x < info.core_size[0]; x++) {
        for (int y = 0; y < info.core_size[1]; y++) {
            for (int z = 0; z < info.core_size[2]; z++) {
                int rx = i * info.stride + x - info.padding;
                int ry = j * info.stride + y - info.padding;
                if (in_bound(rx, ry, info.in_size)) {
                    score[gid[0] + gid[1]]
                    +=
                    input[rx * info.in_size[1] * info.in_size[2] +
                          ry * info.in_size[2] +
                          z]
                    *
                    core[x * info.core_size[1] * info.core_size[2] * info.out_size[2] +
                         y * info.core_size[2] * info.out_size[2] +
                         z * info.out_size[2] +
                         gid[1]];
                }
            }
        }
    }
    
    score[gid[0] + gid[1]] += bi[gid[1]];
}
