//
//  step_m.metal
//  nn
//
//  Created by Liuliet.Lee on 7/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void param_step(device const float &lr,
                       device const float &momentum,
                       device const float *d,
                       device float *v,
                       device float *p,
                       uint i [[ thread_position_in_grid ]])
{
    v[i] = momentum * v[i] + d[i];
    p[i] -= lr * v[i];
}
