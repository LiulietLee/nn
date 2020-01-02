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
    
    private func matrixMul(matrix: Matrix, input: NNArray) {
        let pipeline = Core.pipeline(by: "dense_matrix_mul");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(matrix._data), Core.buffer(input), Core.buffer(&inFeatures)
        )
        
        let gridSize = MTLSizeMake(outFeatures * inFeatures, 1, 1)
        let threadSize = MTLSizeMake(min(outFeatures * inFeatures, 512), 1, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func matrixSum(matrix: Matrix) {
        let pipeline = Core.pipeline(by: "dense_matrix_sum");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(matrix._data), Core.buffer(&inFeatures), Core.buffer(bias), Core.buffer(&relu), Core.buffer(score), Core.buffer(interScore)
        )
        
        let gridSize = MTLSizeMake(outFeatures, 1, 1)
        let threadSize = MTLSizeMake(min(outFeatures, 512), 1, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func forwardWithMetal(_ input: NNArray) {
        let cp: Matrix = param.copy()
        matrixMul(matrix: cp, input: input)
        matrixSum(matrix: cp)
    }
}

extension Dense {
    
    private func backward1(_ da: NNArray, _ delta: NNArray, _ rate: Float) {
        // let row = outFeatures, col = inFeatures
        var rate = rate
        let pipeline = Core.pipeline(by: "dense_backward_1");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&relu), Core.buffer(&rate), Core.buffer(&outFeatures), Core.buffer(&inFeatures), Core.buffer(param._data), Core.buffer(delta), Core.buffer(interScore), Core.buffer(da)
        )
        
        let gridSize = MTLSizeMake(inFeatures, 1, 1)
        let threadSize = MTLSizeMake(min(inFeatures, 512), 1, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func backward2(_ input: NNArray, _ delta: NNArray, _ rate: Float) {
        // let row = outFeatures, col = inFeatures
        var rate = rate
        let pipeline = Core.pipeline(by: "dense_backward_2");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&relu), Core.buffer(&rate), Core.buffer(&inFeatures), Core.buffer(delta), Core.buffer(input), Core.buffer(interScore), Core.buffer(param._data), Core.buffer(bias)
        )
        
        let gridSize = MTLSizeMake(inFeatures * outFeatures, 1, 1)
        let threadSize = MTLSizeMake(min(inFeatures * outFeatures, 512), 1, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func backwardWithMetal(_ input: NNArray, _ delta: NNArray, _ rate: Float) -> NNArray {
        let da = NNArray(input.count, initValue: 0.0)

        backward1(da, delta, rate)
        backward2(input, delta, rate)
        
        return da
    }
}
