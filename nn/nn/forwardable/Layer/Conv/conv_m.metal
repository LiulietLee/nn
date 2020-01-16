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
    int3 core_size; // [depth, width, height]
    int3 in_size;   // [depth, width of input, height of input]
    int3 out_size;  // [count, row, col]
    int batch_size;
    int stride;
    int padding;
};

bool in_bound(int x, int y, int row, int col) {
    return 0 <= x && x < row && 0 <= y && col;
}

kernel void conv_forward(device const conv_layer_info &info,
                         device const float *input,
                         device const float *core,
                         device const float *bi,
                         device const int &input_length,
                         device const int &core_length,
                         device const int &bi_length,
                         device const int &score_length,
                         device float *score,
                         uint3 gid [[ thread_position_in_grid ]])
{
    // gid = [batch, count, row * col]
    int i = gid[2] / info.out_size[2];
    int j = gid[2] % info.out_size[2];
    int index =
    gid[0] * info.out_size[0] * info.out_size[1] * info.out_size[2] +
    gid[1] * info.out_size[1] * info.out_size[2] +
    gid[2];
    
    if (index >= score_length) return;
    
    score[index] = 0.0;
    for (int x = 0; x < info.core_size[1]; x++) {
        for (int y = 0; y < info.core_size[2]; y++) {
            for (int z = 0; z < info.core_size[0]; z++) {
                int rx = i * info.stride + x - info.padding;
                int ry = j * info.stride + y - info.padding;
                if (in_bound(rx, ry, info.in_size[1], info.in_size[2])) {
                    int ii =
                    gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                    z * info.in_size[1] * info.in_size[2] +
                    rx * info.in_size[2] +
                    ry;
                    
                    int ic =
                    gid[1] * info.core_size[0] * info.core_size[1] * info.core_size[2] +
                    z * info.core_size[1] * info.core_size[2] +
                    x * info.core_size[2] +
                    y;
                    
                    if (ii >= input_length || ic >= core_length) continue;
                    
                    score[index] += input[ii] * core[ic];
                }
            }
        }
    }
    
    if ((int)gid[1] >= bi_length) return;
    
    score[index] += bi[gid[1]];
}

kernel void conv_backward_1(device const conv_layer_info &info,
                            device const float *core,
                            device const float *delta,
                            device const int &core_length,
                            device const int &delta_length,
                            device const int &da_length,
                            device float *da,
                            uint3 gid [[thread_position_in_grid ]])
{
    // gid = [batch, depth, input width * height]
    int rx = gid[2] / info.in_size[2];
    int ry = gid[2] % info.in_size[2];
    
    int index =
    gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
    gid[1] * info.in_size[1] * info.in_size[2] +
    rx * info.in_size[2] +
    ry;
    
    if (index >= da_length) {
        return;
    }
    
    for (int c = 0; c < info.out_size[0]; c++) {
        for (int x = 0; x < info.core_size[1]; x++) {
            for (int y = 0; y < info.core_size[2]; y++) {
                if ((rx + info.padding - x) % info.stride == 0 &&
                    (ry + info.padding - y) % info.stride == 0) {
                    int i = (rx + info.padding - x) / info.stride;
                    int j = (ry + info.padding - y) / info.stride;
                    if (in_bound(i, j, info.out_size[1], info.out_size[2])) {
                        int ic =
                        c * info.core_size[0] * info.core_size[1] * info.core_size[2] +
                        gid[1] * info.core_size[1] * info.core_size[2] +
                        x * info.core_size[2] +
                        y;
                        
                        int id =
                        gid[0] * info.out_size[0] * info.out_size[1] * info.out_size[2] +
                        c * info.out_size[1] * info.out_size[2] +
                        i * info.out_size[2] +
                        j;
                        
                        if (ic >= core_length || id >= delta_length) continue;
                        
                        da[index] += core[ic] * delta[id];
                    }
                }
            }
        }
    }
}

kernel void conv_backward_2(device const conv_layer_info &info,
                            device const float *input,
                            device const float *delta,
                            device const bool &need_bias,
                            device const int &input_length,
                            device const int &delta_length,
                            device const int &dbias_length,
                            device const int &dcore_length,
                            device float *dbias,
                            device float *dcore,
                            uint3 gid [[thread_position_in_grid ]])
{   // gid = [batch, count * depth, width * height]
    int c = gid[1] / info.in_size[0];
    int z = gid[1] % info.in_size[0];
    int x = gid[2] / info.core_size[2];
    int y = gid[2] % info.core_size[2];
    float sum = 0.0;
    
    for (int i = 0; i < info.out_size[1]; i++) {
        for (int j = 0; j < info.out_size[2]; j++) {
            int id =
            gid[0] * info.out_size[0] * info.out_size[1] * info.out_size[2] +
            c * info.out_size[1] * info.out_size[2] +
            i * info.out_size[2] +
            j;
            
            if (id >= delta_length) continue;
            
            float cur_delta = delta[id];
            sum += cur_delta;
            
            int rx = i * info.stride + x - info.padding;
            int ry = j * info.stride + y - info.padding;
            if (in_bound(rx, ry, info.in_size[1], info.in_size[2])) {
                int ic =
                gid[0] * info.out_size[0] * info.core_size[0] * info.core_size[1] * info.core_size[2] +
                c * info.core_size[0] * info.core_size[1] * info.core_size[2] +
                z * info.core_size[1] * info.core_size[2] +
                x * info.core_size[2] +
                y;
                
                int ii =
                gid[0] * info.in_size[0] * info.in_size[1] * info.in_size[2] +
                z * info.in_size[1] * info.in_size[2] +
                rx * info.in_size[2] +
                ry;
                
                if (ic >= dcore_length || ii >= input_length) continue;
                
                dcore[ic] += input[ii] * cur_delta;
            }
        }
    }
    
    if (need_bias && z == 0 && gid[2] == 0) {
        int ib = gid[0] * info.out_size[0] + c;
        if (ib >= dbias_length) return;
        dbias[ib] += sum;
    }
}
