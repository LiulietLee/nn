//
//  Conv+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 26/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders
import simd

extension Conv {
    
    struct ConvLayerInfo {
        var coreSize: SIMD3<Int32> // [width, height, depth]
        var inSize: SIMD3<Int32>   // [width of input, height of input, depth]
        var outSize: SIMD3<Int32>  // [row, col, count]
        var stride: Int
        var padding: Int
        
        init(_ obj: Conv, input: NNArray) {
            coreSize = SIMD3<Int32>(Int32(obj.width), Int32(obj.height), Int32(obj.depth))
            inSize = SIMD3<Int32>(Int32(input.d[0]), Int32(input.d[1]), Int32(input.d[2]))
            outSize = SIMD3<Int32>(Int32(obj.row), Int32(obj.col), Int32(obj.count))
            stride = obj.step
            padding = obj.padding
        }
    }
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "conv_forward");
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(convCore), Core.buffer(bias), Core.buffer(score)
        )
        
        let gridSize = MTLSizeMake(row * col, count, 1)
        let w = pipeline.threadExecutionWidth
        let h = pipeline.maxTotalThreadsPerThreadgroup / w
        let threadSize = MTLSizeMake(min(row * col, w), min(count, h), 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }

}
