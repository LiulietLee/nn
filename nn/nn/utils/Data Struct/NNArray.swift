//
//  NNArray.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class NNArray {
    public typealias Pointer = LLVector<Float>
    
    public let data: Pointer!
    var d = [Int]()
    var acci = [Int]()
    
    public init() {
        data = Pointer()
    }
    
    public init(_ d: Int..., initValue: Float = 0.0) {
        data = Pointer(repeaing: initValue, count: d.reduce(1, *))
        self.d = d
        setAcci()
    }
    
    private init(_ data: Pointer, d: [Int]) {
        self.data = data
        self.d = d
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
