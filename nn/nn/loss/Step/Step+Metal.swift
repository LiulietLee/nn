//
//  Step+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 7/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

func stepWithMetal(lr: Float, momentum: Float, d: NNArray, v: NNArray, p: NNArray) {
    let pipeline = Core.pipeline(by: "param_step");
    let queue = Core.queue()
    
    let commandBuffer = queue.makeCommandBuffer()!
    var lr = lr
    var momentum = momentum
    Core.encode(
        commandBuffer: commandBuffer,
        pipeline: pipeline,
        buffers: Core.buffer(&lr), Core.buffer(&momentum), Core.buffer(d), Core.buffer(v), Core.buffer(p),
        grid: [v.count, 1, 1],
        thread: [min(v.count, 512), 1, 1]
    )
            
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
}
