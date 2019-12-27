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
                    score[gid[0] * info.out_size[2] + gid[1]]
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
    
    score[gid[0] * info.out_size[2] + gid[1]] += bi[gid[1]];
}

kernel void conv_backward_1(device const conv_layer_info &info,
                            device const float *core,
                            device const float *delta,
                            device const float &rate,
                            device float *da,
                            uint2 gid [[thread_position_in_grid ]])
{
    int rx = gid[0] / info.in_size[1];
    int ry = gid[0] % info.in_size[1];
    
    da[gid[0] * info.in_size[2] + gid[1]] = 0.0;
    for (int c = 0; c < info.out_size[2]; c++) {
        for (int x = 0; x < info.core_size[0]; x++) {
            for (int y = 0; y < info.core_size[1]; y++) {
                if ((rx + info.padding - x) % info.stride == 0 &&
                    (ry + info.padding - y) % info.stride == 0) {
                    int i = (rx + info.padding - x) / info.stride;
                    int j = (ry + info.padding - y) / info.stride;
                    if (in_bound(i, j, info.out_size)) {
                        //da[rx, ry, z] += convCore[x, y, z, c] * delta[i, j, c] * rate
                        da[rx * info.in_size[1] * info.in_size[2] +
                           ry * info.in_size[2] +
                           gid[1]]
                        +=
                        core[x * info.core_size[1] * info.core_size[2] * info.out_size[2] +
                             y * info.core_size[2] * info.out_size[2] +
                             gid[1] * info.out_size[2] +
                             c]
                        *
                        delta[i * info.out_size[1] * info.out_size[2] +
                              j * info.out_size[2] +
                              c]
                        * rate;
                    }
                }
            }
        }
    }
}

kernel void conv_backward_2(device const conv_layer_info &info,
                            device const float *input,
                            device const float *delta,
                            device const float &rate,
                            device const bool &need_bias,
                            device float *bi,
                            device float *core,
                            uint3 gid [[thread_position_in_grid ]])
{   // gid := (width * height, depth, count)
    int x = gid[0] / info.core_size[1];
    int y = gid[0] % info.core_size[1];
    float sum = 0.0;
    
    for (int i = 0; i < info.out_size[0]; i++) {
        for (int j = 0; j < info.out_size[1]; j++) {
            float cur_delta = delta[i * info.out_size[1] * info.out_size[2] +
                                    j * info.out_size[2] +
                                    gid[2]];
            sum += cur_delta;
            
            int rx = i * info.stride + x - info.padding;
            int ry = j * info.stride + y - info.padding;
            if (in_bound(rx, ry, info.in_size)) {
                core[gid[0] * info.core_size[2] * info.out_size[2] +
                     gid[1] * info.out_size[2] +
                     gid[2]]
                -=
                input[rx * info.in_size[1] * info.in_size[2] +
                      ry * info.in_size[2] +
                      gid[1]]
                * cur_delta * rate;
            }
        }
    }
    
    if (need_bias && gid[0] == 0 && gid[1] == 0) {
        bi[gid[2]] -= sum * rate;
    }
}
