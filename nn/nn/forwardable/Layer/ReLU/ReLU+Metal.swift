//
//  ReLU+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 28/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

extension ReLU {
    
    func forwardWithMetal(_ input: NNArray) -> NNArray {
        let pipeline = Core.pipeline(by: "relu_forward");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(input), Core.buffer(score),
            grid: [input.count, 1, 1],
            thread: [min(input.count, 512), 1, 1]
        )
                
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return score
    }
}

extension ReLU {
    
    func backwardWithMetal(_ input: NNArray, _ delta: NNArray) -> NNArray {
        let da = NNArray(input.count)

        let pipeline = Core.pipeline(by: "relu_backward");
        let queue = Core.queue()
        
        let commandBuffer = queue.makeCommandBuffer()!
        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(input), Core.buffer(delta), Core.buffer(da),
            grid: [input.count, 1, 1],
            thread: [min(input.count, 512), 1, 1]
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return da
    }
}
