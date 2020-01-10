//
//  Dense+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 22/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

extension Dense {
    
    private func matrixMul(matrix: NNArray, input: NNArray) {
        let pipeline = Core.pipeline(by: "dense_matrix_mul");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(outFeatures, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(inFeatures, pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(input), Core.buffer(&outFeatures),  Core.buffer(&inFeatures), Core.buffer(matrix),
            grid: [batchSize, outFeatures, inFeatures],
            thread: [w, h, d]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func matrixSum(matrix: NNArray) {
        let pipeline = Core.pipeline(by: "dense_matrix_sum");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(outFeatures, pipeline.maxTotalThreadsPerThreadgroup / w)
        
        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(matrix), Core.buffer(&inFeatures), Core.buffer(&outFeatures), Core.buffer(bias), Core.buffer(&relu), Core.buffer(score), Core.buffer(interScore),
            grid: [batchSize, outFeatures, 1],
            thread: [w, h, 1]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func forwardWithMetal(_ input: NNArray) {
        let cp = NNArray(batchSize, outFeatures, inFeatures)
        for i in 0..<batchSize {
            memcpy(
                cp.subArray(at: i).data.pointer,
                param.data.pointer,
                MemoryLayout<Float>.stride * outFeatures * inFeatures
            )
        }
        matrixMul(matrix: cp, input: input)
        matrixSum(matrix: cp)
    }
}

extension Dense {
    
    private func backward1(_ da: NNArray, _ delta: NNArray) {
        // let row = outFeatures, col = inFeatures
        let pipeline = Core.pipeline(by: "dense_backward_1");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(inFeatures, pipeline.maxTotalThreadsPerThreadgroup / w)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&relu), Core.buffer(&outFeatures), Core.buffer(&inFeatures), Core.buffer(param), Core.buffer(delta), Core.buffer(interScore), Core.buffer(da),
            grid: [batchSize, inFeatures, 1],
            thread: [w, h, 1]
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func backward2(_ input: NNArray, _ delta: NNArray) {
        // let row = outFeatures, col = inFeatures
        let pipeline = Core.pipeline(by: "dense_backward_2");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(outFeatures, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(inFeatures, pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&relu), Core.buffer(&outFeatures),  Core.buffer(&inFeatures), Core.buffer(delta), Core.buffer(input), Core.buffer(interScore), Core.buffer(dparam), Core.buffer(dbias),
            grid: [batchSize, outFeatures, inFeatures],
            thread: [w, h, d]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func backwardWithMetal(_ input: NNArray, _ delta: NNArray) -> NNArray {
        let da = NNArray(input.count)

        backward1(da, delta)
        backward2(input, delta)
        
        return da
    }
}
