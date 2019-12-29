//
//  Core.swift
//  nn
//
//  Created by Liuliet.Lee on 22/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

public class Core {
    
    public static var device: MTLDevice? = nil {
        didSet {
            if device != nil {
                library = device!.makeDefaultLibrary()!
            }
        }
    }
    
    static var library: MTLLibrary!
    
    static func function(_ name: String) -> MTLFunction {
        return library.makeFunction(name: name)!
    }
    
    static func pipeline(by functionName: String) -> MTLComputePipelineState {
        return try! device!.makeComputePipelineState(function: function(functionName))
    }
    
    static func queue() -> MTLCommandQueue {
        return device!.makeCommandQueue()!
    }
    
    static func encode(commandBuffer: MTLCommandBuffer, pipeline: MTLComputePipelineState, buffers: MTLBuffer...) -> MTLComputeCommandEncoder {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for i in 0..<buffers.count {
            encoder.setBuffer(buffers[i], offset: 0, index: i)
        }
        return encoder
    }
    
    static func buffer<T>(_ data: inout T, count: Int = 1) -> MTLBuffer {
        return device!.makeBuffer(
            bytes: &data,
            length: MemoryLayout<T>.stride * count,
            options: .storageModeShared
        )!
    }
    
    static func buffer<T>(_ vec: LLVector<T>) -> MTLBuffer {
        return device!.makeBuffer(
            bytesNoCopy: vec.pointer,
            length: vec.byteSize,
            options: .storageModeShared,
            deallocator: nil
        )!
    }
    
    static func buffer(_ arr: NNArray) -> MTLBuffer {
        return buffer(arr.data)
    }
}
