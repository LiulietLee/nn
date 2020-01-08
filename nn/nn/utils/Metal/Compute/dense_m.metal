//
//  dense_m.metal
//  nn
//
//  Created by Liuliet.Lee on 22/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void dense_matrix_mul(device float *matrix,
                             device const float *input,
                             device const int &col,
                             uint2 gid [[ thread_position_in_grid ]])
{
    uint index = gid[0] * col + gid[1];
    matrix[index] *= input[index % col];
}

kernel void dense_matrix_sum(device const float *matrix,
                             device const int &col,
                             device const float *bi,
                             device const bool &need_relu,
                             device float *score,
                             device float *inter_score,
                             uint index [[ thread_position_in_grid ]])
{
    inter_score[index] = 0.0;
    
    for (int i = 0; i < col; i++) {
        inter_score[index] += matrix[index * col + i];
    }

    score[index] = inter_score[index];
    if (need_relu && inter_score[index] < 0.0) {
        score[index] *= 0.001;
    }
    
    score[index] += bi[index];
}

kernel void dense_backward_1(device const bool &need_relu,
                             device const int &row,
                             device const int &col,
                             device const float *matrix,
                             device const float *delta,
                             device const float *inter_score,
                             device float *da,
                             uint j [[ thread_position_in_grid ]])
{
    da[j] = 0.0;
    for (int i = 0; i < row; i++) {
        float d = delta[i] * matrix[i * col + j];
        if (need_relu && inter_score[i] < 0.0) {
            da[j] += d * 0.001;
        } else {
            da[j] += d;
        }
    }
}

kernel void dense_backward_2(device const bool &need_relu,
                             device const int &col,
                             device const float *delta,
                             device const float *input,
                             device const float *inter_score,
                             device float *dparam,
                             device float *dbias,
                             uint2 gid [[ thread_position_in_grid ]])
{
    int i = gid[0];
    int j = gid[1];
    
    if (j == 0) {
        dbias[i] += delta[i];
    }
    
    float d = delta[i] * input[j];
    if (need_relu && inter_score[i] < 0.0) {
        dparam[i * col + j] += d * 0.001;
    } else {
        dparam[i * col + j] += d;
    }
}
