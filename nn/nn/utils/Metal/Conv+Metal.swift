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
}

extension Conv {
    
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
        let w = min(row * col, pipeline.threadExecutionWidth)
        let h = min(count, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadSize = MTLSizeMake(w, h, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }
}

extension Conv {
    
    private func backward1(_ da: NNArray, input: NNArray, delta: NNArray, rate: Float) {
        var rate = rate
        let pipeline = Core.pipeline(by: "conv_backward_1");
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(convCore), Core.buffer(delta), Core.buffer(&rate), Core.buffer(da)
        )
        
        let gridSize = MTLSizeMake(row * col, depth, 1)
        let w = min(row * col, pipeline.threadExecutionWidth)
        let h = min(depth, pipeline.maxTotalThreadsPerThreadgroup / w)
        let threadSize = MTLSizeMake(w, h, 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func backward2(input: NNArray, delta: NNArray, rate: Float) {
        var rate = rate
        let pipeline = Core.pipeline(by: "conv_backward_2");
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let encoder = Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(delta), Core.buffer(&rate), Core.buffer(&needBias), Core.buffer(bias), Core.buffer(convCore)
        )
        
        let gridSize = MTLSizeMake(width * height, depth, count)
        let w = min(width * height, pipeline.threadExecutionWidth)
        let h = min(depth, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(count, max(1, pipeline.maxTotalThreadsPerThreadgroup / w / h))
        let threadSize = MTLSizeMake(w, h, d)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func backwardWithMetal(_ da: NNArray, _ input: NNArray, _ delta: NNArray, _ rate: Float) -> NNArray {
        backward1(da, input: input, delta: delta, rate: rate)
        backward2(input: input, delta: delta, rate: rate)
        return da
    }
}
