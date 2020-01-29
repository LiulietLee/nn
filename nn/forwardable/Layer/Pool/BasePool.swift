//
//  BasePool.swift
//  nn
//
//  Created by Liuliet.Lee on 22/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public class BasePool: BaseLayer {
    
    var padding = 0
    var width = 0
    var height = 0
    var step = 0
    
    var row = 0
    var col = 0
    
    /**
     - parameter width: Size of the convolving kernel.
     - parameter height: Ignore this.
     - parameter step: Stride for convolution.
     - parameter padding: Zero-padding added to both sides of the input.
     */
    public init(_ width: Int, _ height: Int = -1, step: Int = 1, padding: Int = 0) {
        self.width = width
        self.height = height <= 0 ? width : height
        self.step = step
        self.padding = padding
    }
    
    struct PoolingLayerInfo {
        var coreSize: SIMD2<Int32>
        var outSize: SIMD2<Int32>
        var inSize: SIMD3<Int32>
        var stride: Int32
        var padding: Int32
        var batchSize: Int32
        
        init(_ obj: BasePool, input: NNArray) {
            coreSize = SIMD2<Int32>(Int32(obj.width), Int32(obj.height))
            outSize = SIMD2<Int32>(Int32(obj.row), Int32(obj.col))
            inSize = SIMD3<Int32>(Int32(input.d[1]), Int32(input.d[2]), Int32(input.d[3]))
            stride = Int32(obj.step)
            padding = Int32(obj.padding)
            batchSize = Int32(obj.batchSize)
        }
    }
}
