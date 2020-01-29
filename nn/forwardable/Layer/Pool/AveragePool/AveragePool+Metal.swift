//
//  AveragePool+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 22/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

extension AveragePool {
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "averagepool_forward");
        let queue = Core.queue()
        var info = PoolingLayerInfo(self, input: input)

        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(input.d[1], pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(row * col, pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(score),
            grid: [batchSize, input.d[1], row * col],
            thread: [w, h, d]
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }
}

extension AveragePool {
    
    func backwardWithMetal(_ da: NNArray, _ input: NNArray, _ delta: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "averagepool_backward");
        let queue = Core.queue()
        var info = PoolingLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(input.d[0], pipeline.threadExecutionWidth)
        let h = min(input.d[1], pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(input.d[2] * input.d[3], pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(delta), Core.buffer(da),
            grid: [input.d[0], input.d[1], input.d[2] * input.d[3]],
            thread: [w, h, d]
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return da
    }
}
