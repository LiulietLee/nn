//
//  NNArray.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class NNArray: NSObject {
    public typealias Pointer = LLVector<Float>
    
    public var data: Pointer!
    var d = [Int]()
    var acci = [Int]()
    
    public override var description: String { return "\(map { $0 })" }
    
    public override init() {
        data = Pointer()
    }
    
    public init(_ d: Int..., initValue: Float = 0.0) {
        data = Pointer(repeaing: initValue, count: d.reduce(1, *))
        self.d = d
        super.init()
        setAcci()
    }
    
    public init(_ d: [Int], initValue: Float = 0.0) {
        data = Pointer(repeaing: initValue, count: d.reduce(1, *))
        self.d = d
        super.init()
        setAcci()
    }
    
    public init(_ data: Pointer, d: [Int]) {
        self.data = data
        self.d = d
        super.init()
        setAcci()
    }

    init(_ data: [Float], d: [Int] = []) {
        self.data = Pointer()
        self.data.append(contentsOf: data)
        if d == [] {
            self.d = [data.count]
        } else {
            self.d = d
        }
        super.init()
        setAcci()
    }

    @discardableResult
    public func dim(_ d: Int...) -> NNArray {
        return dim(d)
    }
    
    @discardableResult
    public func dim(_ d: [Int]) -> NNArray {
        precondition(d.reduce(1, *) == count)
        self.d = d
        setAcci()
        return self
    }
    
    public func copy() -> NNArray {
        return NNArray(data.copy(), d: d)
    }
    
    private func setAcci() {
        acci = Array.init(repeating: 1, count: d.count)
        for i in (0..<d.count - 1).reversed() {
            acci[i] = acci[i + 1] * d[i + 1]
        }
    }
    
    private func getAddr(_ index: [Int]) -> Int {
        precondition(index.count == d.count)
        return zip(index, acci).map { $0.0 * $0.1 }.reduce(0, +)
    }
    
    public subscript(index: Int...) -> Float {
        get {
            return data[getAddr(index)]
        }
        set(newValue) {
            data[getAddr(index)] = newValue
        }
    }
    
    public subscript(index: Int) -> Float {
        get {
            return data[index]
        }
        set(newValue) {
            data[index] = newValue
        }
    }
    
    public func subArray(at index: Int..., length: Int = 0, d: [Int] = []) -> NNArray {
        var idx = index
        while idx.count < self.d.count {
            idx.append(0)
        }
        let addr = getAddr(idx)
        let ptr = data.pointer.advanced(by: addr * MemoryLayout<Float>.stride)
        let len = length == 0 ? acci[index.count - 1] : length
        let vec = LLVector<Float>(ptr, len, len)
        vec.freeable = false
        var d = d
        if d.isEmpty {
            d = self.d
            for i in 0..<index.count {
                d[i] = 1
            }
        }
        return NNArray(vec, d: d)
    }
}

extension NNArray: Sequence {
    public struct Iterator: IteratorProtocol {
        var current = 0
        var pointer: Pointer
        var length: Int
        
        init(_ pointer: Pointer, _ length: Int) {
            self.pointer = pointer
            self.length = length
        }
        
        mutating public func next() -> Float? {
            if current < length {
                defer {
                    current += 1
                }
                return pointer[current]
            } else {
                return nil
            }
        }
    }
    
    public __consuming func makeIterator() -> Iterator {
        return Iterator(data, data.count)
    }
}

extension NNArray: RandomAccessCollection {
    public var startIndex: Int { return 0 }
    public var endIndex: Int { return data.count }
}

extension NNArray: MutableCollection {}

extension NNArray {
    /**
     - Important: Every shape of each buffer need to be the same height and width.
     */
    public static func concat(_ buffers: [NNArray], d: [Int] = []) -> NNArray {
        var d = d
        if d.isEmpty {
            d = [buffers.reduce(0, { (res, buf) -> Int in
                return res + buf.d.reduce(1, *)
            })]
        }
        let res = NNArray(d)
        var src = res.data.pointer!

        for buf in buffers {
            memcpy(src, buf.data.pointer, buf.data.byteCount)
            src = src.advanced(by: buf.data.byteCount)
        }
        
        return res
    }
}

extension NNArray {
    public func indexOfMax() -> Int {
        var maxi = 0
        for i in 0..<count {
            if self[maxi] < self[i] {
                maxi = i
            }
        }
        return maxi
    }
}
