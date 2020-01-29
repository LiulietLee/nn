//
//  step_m.metal
//  nn
//
//  Created by Liuliet.Lee on 7/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void param_step(device const int &batch,
                       device const float &lr,
                       device const float &momentum,
                       device const int &count,
                       device const float *d,
                       device float *m,
                       device float *v,
                       device float *p,
                       uint i [[ thread_position_in_grid ]])
{
    for (int j = 0; j < batch; j++) {
        int idx = j * count + i;
        m[idx] = 0.9 * m[idx] + (1 - 0.9) * d[idx];
        v[idx] = momentum * v[idx] + (1 - 0.9) * d[idx] * d[idx];
        p[i] -= lr * m[idx] / (sqrt(v[idx]) + 0.0000001);
    }
}
