//
//  metal_utils.h
//  nn
//
//  Created by Liuliet.Lee on 19/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

#ifndef metal_utils_h
#define metal_utils_h

bool in_bound(int x, int y, int row, int col);

struct pooling_layer_info {
    int2 core_size;
    int2 out_size;
    int3 in_size;
    int stride;
    int padding;
    int batch_size;
};

#endif /* metal_utils_h */
