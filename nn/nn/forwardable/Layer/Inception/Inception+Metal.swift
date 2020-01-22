//
//  Inception+Metal.swift
//  nn
//
//  Created by Liuliet.Lee on 20/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

extension Inception {
    
    func sumInterDelta() -> NNArray {
        let pipeline = Core.pipeline(by: "sum_inter_delta");
        let queue = Core.queue()
        
        let da = NNArray(interDelta[0].d)

        let concated = NNArray.concat(interDelta)
        var coreSize = core.count
        var length = da.count
        
        let commandBuffer = queue.makeCommandBuffer()!
        Core.encode(
            commandBuffer: commandBuffer,
            pipeline: pipeline,
            buffers: Core.buffer(concated), Core.buffer(&coreSize), Core.buffer(&length), Core.buffer(da),
            grid: [length, 1, 1],
            thread: [min(length, 512), 1, 1]
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return da
    }
}
