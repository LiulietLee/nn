//
//  Step+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 7/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

func stepWithMetal(batch: Int, lr: Float, momentum: Float, d: NNArray, m: NNArray, v: NNArray, p: NNArray) {
    let pipeline = Core.pipeline(by: "param_step");
    let queue = Core.queue()
    
    let commandBuffer = queue.makeCommandBuffer()!
    var lr = lr
    var momentum = momentum
    var count = p.count
    var batch = batch
    let w = min(p.count, pipeline.threadExecutionWidth)
    
    Core.encode(
        commandBuffer: commandBuffer,
        pipeline: pipeline,
        buffers: Core.buffer(&batch), Core.buffer(&lr), Core.buffer(&momentum), Core.buffer(&count),  Core.buffer(d), Core.buffer(m), Core.buffer(v), Core.buffer(p),
        grid: [p.count, 1, 1],
        thread: [w, 1, 1]
    )
            
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}
