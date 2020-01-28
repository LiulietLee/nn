//
//  dense_m.metal
//  nn
//
//  Created by Liuliet.Lee on 22/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void dense_matrix_mul(device const float *input,
                             device const int &out_features,
                             device const int &in_features,
                             device float *matrix,
                             uint3 gid [[ thread_position_in_grid ]])
{
    // gid = [batch, outFeatures, inFeatures]
    matrix[gid[0] * out_features * in_features +
           gid[1] * in_features +
           gid[2]]
    *=
    input[gid[0] * in_features +
          gid[2]];
}

kernel void dense_matrix_sum(device const float *matrix,
                             device const int &in_features,
                             device const int &out_features,
                             device const float *bi,
                             device const bool &need_relu,
                             device float *score,
                             device float *inter_score,
                             uint2 gid [[ thread_position_in_grid ]])
{
    // gid = [batch, outFeatures]
    int start = gid[0] * in_features * out_features + gid[1] * in_features;
    int index = gid[0] * out_features + gid[1];
    inter_score[index] = 0.0;
    
    for (int i = 0; i < in_features; i++) {
        inter_score[index] += matrix[start + i];
    }

    score[index] = inter_score[index];
    if (need_relu && inter_score[index] < 0.0) {
        score[index] *= 0.001;
    }
    
    score[index] += bi[gid[1]];
}

kernel void dense_backward_1(device const bool &need_relu,
                             device const int &out_features,
                             device const int &in_features,
                             device const float *matrix,
                             device const float *delta,
                             device const float *inter_score,
                             device float *da,
                             uint2 gid [[ thread_position_in_grid ]])
{
    // gid = [batch, inFeatures]
    int index = gid[0] * in_features + gid[1];
    da[index] = 0.0;
    for (int i = 0; i < out_features; i++) {
        float d = delta[gid[0] * out_features + i] * matrix[i * in_features + gid[1]];
        
        if (need_relu && inter_score[gid[0] * out_features + i] < 0.0) {
            da[index] += d * 0.001;
        } else {
            da[index] += d;
        }
    }
}

kernel void dense_backward_2(device const bool &need_relu,
                             device const int &out_features,
                             device const int &in_features,
                             device const float *delta,
                             device const float *input,
                             device const float *inter_score,
                             device float *dparam,
                             device float *dbias,
                             uint3 gid [[ thread_position_in_grid ]])
{
    // gid = [batch, outFeatures, inFeatures]
    int batch = gid[0];
    int i = gid[1];
    int j = gid[2];
    
    if (j == 0) {
        dbias[batch * out_features + i] += delta[batch * out_features + i];
    }
    
    float d = delta[batch * out_features + i] * input[batch * in_features + j];
    int index = batch * in_features * out_features + i * in_features + j;
    if (need_relu && inter_score[batch * out_features + i] < 0.0) {
        dparam[index] += d * 0.001;
    } else {
        dparam[index] += d;
    }
}
