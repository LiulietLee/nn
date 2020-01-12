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
        var batchSize: Int32
        var stride: Int32
        var padding: Int32
        
        init(_ obj: Conv, input: NNArray) {
            coreSize = SIMD3<Int32>(Int32(obj.depth), Int32(obj.width), Int32(obj.height))
            inSize = SIMD3<Int32>(Int32(input.d[1]), Int32(input.d[2]), Int32(input.d[3]))
            outSize = SIMD3<Int32>(Int32(obj.count), Int32(obj.row), Int32(obj.col))
            batchSize = Int32(obj.batchSize)
            stride = Int32(obj.step)
            padding = Int32(obj.padding)
        }
    }
}

extension Conv {
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "conv_forward")
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(count, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(row * col, pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(convCore), Core.buffer(bias), Core.buffer(score),
            grid: [batchSize, count, row * col],
            thread: [w, h, d]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }
}

extension Conv {
    
    private func backward1(_ da: NNArray, input: NNArray, delta: NNArray) {
        let pipeline = Core.pipeline(by: "conv_backward_1");
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(depth, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(input.d[2] * input.d[3], pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(convCore), Core.buffer(delta), Core.buffer(da),
            grid: [batchSize, depth, input.d[2] * input.d[3]],
            thread: [w, h, d]
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    private func backward2(input: NNArray, delta: NNArray) {
        let pipeline = Core.pipeline(by: "conv_backward_2");
        let queue = Core.queue()
        var info = ConvLayerInfo(self, input: input)
        
        let commandBuffer = queue.makeCommandBuffer()!
        let w = min(batchSize, pipeline.threadExecutionWidth)
        let h = min(count * depth, pipeline.maxTotalThreadsPerThreadgroup / w)
        let d = min(width * height, pipeline.maxTotalThreadsPerThreadgroup / w / h)

        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(&info), Core.buffer(input), Core.buffer(delta), Core.buffer(&needBias), Core.buffer(dbias), Core.buffer(dcore),
            grid: [batchSize, count * depth, width * height],
            thread: [w, h, d]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    func backwardWithMetal(_ da: NNArray, _ input: NNArray, _ delta: NNArray) -> NNArray {
        backward1(da, input: input, delta: delta)
        backward2(input: input, delta: delta)
        return da
    }
}
