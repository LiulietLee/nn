//
//  relu_m.metal
//  nn
//
//  Created by Liuliet.Lee on 28/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void relu_forward(device const float *input,
                         device float *score,
                         uint index [[ thread_position_in_grid ]])
{
    score[index] = max(input[index] * 0.001, input[index]);
}

kernel void relu_backward(device const float *input,
                          device const float *delta,
                          device float *da,
                          uint index [[ thread_position_in_grid ]])
{
    da[index] = delta[index];
    if (input[index] < 0.0) {
        da[index] *= 0.001;
    }
}
