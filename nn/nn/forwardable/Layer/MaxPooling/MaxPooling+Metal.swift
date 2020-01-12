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
        var stride: Int32
        var batchSize: Int32
        
        init(_ obj: MaxPooling, input: NNArray) {
            coreSize = SIMD2<Int32>(Int32(obj.width), Int32(obj.height))
            outSize = SIMD2<Int32>(Int32(obj.row), Int32(obj.col))
            inSize = SIMD3<Int32>(Int32(input.d[1]), Int32(input.d[2]), Int32(input.d[3]))
            stride = Int32(obj.step)
            batchSize = Int32(obj.batchSize)
        }
    }
}

extension MaxPooling {
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "maxpooling_forward");
        let queue = Core.queue()
        var info = PoolingLayerInfo(self, input: input)

        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(input.d[1], pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(row * col, max(1, pipeline.maxTotalThreadsPerThreadgroup / w / h))

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(switches), Core.buffer(score),
            grid: [batchSize, input.d[1], row * col],
            thread: [w, h, d]
        )

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
        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(switches), Core.buffer(delta), Core.buffer(da),
            grid: [switches.count, 1, 1],
            thread: [min(switches.count, 512), 1, 1]
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return da
    }
}
