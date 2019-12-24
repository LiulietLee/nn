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
                             device const int &inter_col,
                             uint index [[ thread_position_in_grid ]])
{
    matrix[index] *= input[index % inter_col];
}

kernel void dense_matrix_sum(device const float *matrix,
                             device const int &col,
                             device const float *bi,
                             device const bool &need_relu,
                             device float *score,
                             device float *inter_score,
                             uint index [[ thread_position_in_grid ]])
{
    inter_score[index] = bi[index];
    
    for (int i = 0; i < col; i++) {
        inter_score[index] += matrix[index * col + i];
    }

    if (need_relu && inter_score[index] < 0.0) {
        score[index] = 0.0;
    } else {
        score[index] = inter_score[index];
    }
}

kernel void dense_backward_1(device const bool &need_relu,
                             device const float &rate,
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
        if (need_relu) {
            da[j] += (inter_score[i] >= 0.0 ? 1.0 : 0.0) * delta[i] * matrix[i * col + j] * rate;
        } else {
            da[j] += delta[i] * matrix[i * col + j] * rate;
        }
    }
}

kernel void dense_backward_2(device const bool &need_relu,
                             device const float &rate,
                             device const int &col,
                             device const float *delta,
                             device const float *input,
                             device const float *inter_score,
                             device float *matrix,
                             device float *bi,
                             uint index [[ thread_position_in_grid ]])
{
    int i = index / col;
    int j = index % col;
    
    if (j == 0) {
        bi[i] -= delta[i] * rate;
    }
    
    if (need_relu) {
        matrix[index] -= (inter_score[i] >= 0.0 ? 1.0 : 0.0) * delta[i] * input[j] * rate;
    } else {
        matrix[index] -= delta[i] * input[j] * rate;
    }
}
