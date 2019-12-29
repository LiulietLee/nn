//
//  MaxPooling+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 29/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

extension MaxPooling {
    struct PoolingLayerInfo {
        var coreSize: SIMD2<Int32>
        var outSize: SIMD2<Int32>
        var inSize: SIMD3<Int32>
        var stride: Int
        
        init(_ obj: MaxPooling, input: NNArray) {
            coreSize = SIMD2<Int32>(Int32(obj.width), Int32(obj.height))
            outSize = SIMD2<Int32>(Int32(obj.row), Int32(obj.col))
            inSize = SIMD3<Int32>(Int32(input.d[0]), Int32(input.d[1]), Int32(input.d[2]))
            stride = obj.step
        }
    }
}

extension MaxPooling {
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "maxpooling_forward");
        let queue = Core.queue()
        var info = PoolingLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(switches), Core.buffer(score)
        )
        
        let gridSize = MTLSizeMake(row, col, input.d[2])
        let w = min(row, pipeline.threadExecutionWidth)
        let h = min(col, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(input.d[2], max(1, pipeline.maxTotalThreadsPerThreadgroup / w / h))
        let threadSize = MTLSizeMake(w, h, d)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }
}

extension MaxPooling {
    
    func backwardWithMetal(_ input: NNArray, _ delta: NNArray) -> NNArray {
        let da = NNArray(input.count)
        
        let pipeline = Core.pipeline(by: "maxpooling_backward");
        let queue = Core.queue()
        var info = PoolingLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(switches), Core.buffer(delta), Core.buffer(da)
        )
        
        let gridSize = MTLSizeMake(switches.count, 1, 1)
        let threadSize = MTLSizeMake(min(switches.count, 512), 1, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return da
    }
}
