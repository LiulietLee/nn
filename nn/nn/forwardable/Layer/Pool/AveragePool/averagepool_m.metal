//
//  averagepool_m.metal
//  nn
//
//  Created by Liuliet.Lee on 22/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

#include "../../../../utils/Metal/metal_utils.h"

kernel void averagepool_forward(device const pooling_layer_info &info,
                                device const float *input,
                                device float *score,
                                uint3 gid [[ thread_position_in_grid ]])
{
    int batch = gid[0];
    int k = gid[1];
    int i = gid[2] / info.out_size[1];
    int j = gid[2] % info.out_size[1];
    
    int ri = i * info.stride;
    int rj = j * info.stride;
    
    float sum = 0.0;
    
    for (int x = 0; x < info.core_size[0]; x++) {
        for (int y = 0; y < info.core_size[1]; y++) {
            int rx = x + ri;
            int ry = y + rj;
            
            sum += input[batch * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                         k * info.in_size[1] * info.in_size[2] +
                         rx * info.in_size[2] +
                         ry];
        }
    }
    
    score[batch * info.in_size[0] * info.out_size[0] * info.out_size[1] +
          k * info.out_size[0] * info.out_size[1] +
          i * info.out_size[1] +
          j]
    = sum / (float)(info.core_size[0] * info.core_size[1]);
}

kernel void averagepool_backward(device const pooling_layer_info &info,
                                 device const float *delta,
                                 device float *da,
                                 uint3 gid [[ thread_position_in_grid ]])
{
    int batch = gid[0];
    int k = gid[1];
    int i = gid[2] / info.in_size[1];
    int j = gid[2] % info.in_size[1];
    
    int ri = i / info.core_size[0];
    int rj = j / info.core_size[1];
    
    da[batch * info.in_size[0] * info.in_size[0] * info.in_size[1] +
       k * info.in_size[0] * info.in_size[1] +
       i * info.in_size[1] +
       j]
    =
    delta[batch * info.in_size[0] * info.out_size[0] * info.out_size[1] +
          k * info.out_size[0] * info.out_size[1] +
          ri * info.out_size[1] +
          rj]
    / (float)(info.core_size[0] * info.core_size[1]);
}
