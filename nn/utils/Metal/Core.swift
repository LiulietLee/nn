//
//  Core.swift
//  nn
//
//  Created by Liuliet.Lee on 22/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

/**
 Metal Core.
 */
public class Core {
    
    public static var device: MTLDevice? = nil {
        didSet {
            if device != nil {
                library = device!.makeDefaultLibrary()!
            }
        }
    }
    
    static var library: MTLLibrary!
    
    static var functionMap = [String: MTLFunction]()
    
    static let mutex = DispatchSemaphore(value: 1)
    
    static func function(_ name: String) -> MTLFunction {
        mutex.wait()
        defer {
            mutex.signal()
        }
        if let function = functionMap[name] {
            return function
        } else {
            let function = library.makeFunction(name: name)!
            functionMap[name] = function
            return function
        }
    }
    
    static func pipeline(by functionName: String) -> MTLComputePipelineState {
        return try! device!.makeComputePipelineState(function: function(functionName))
    }
    
    static func queue() -> MTLCommandQueue {
        return device!.makeCommandQueue()!
    }
    
    @discardableResult
    static func encode(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        buffers: MTLBuffer...,
        grid: [Int],
        thread: [Int]
    ) -> MTLComputeCommandEncoder {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipeline)
        for i in 0..<buffers.count {
            encoder.setBuffer(buffers[i], offset: 0, index: i)
        }
        let gridSize = MTLSizeMake(grid[0], grid[1], grid[2])
        let threadSize = MTLSizeMake(thread[0], thread[1], thread[2])
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadSize)
        encoder.endEncoding()

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
    
    static func startCapture() {
        let captureManager = MTLCaptureManager.shared()
        let captureDescriptor = MTLCaptureDescriptor()
        captureDescriptor.captureObject = Core.device
        do {
            try captureManager.startCapture(with: captureDescriptor)
        } catch {
            fatalError("error when trying to capture: \(error)")
        }
    }

    static func stopCapture() {
        let captureManager = MTLCaptureManager.shared()
        captureManager.stopCapture()
    }
}
