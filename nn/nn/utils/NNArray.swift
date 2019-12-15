//
//  NNArray.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class NNArray: LLVector<Float> {
    var d = [Int]()
    var acci = [Int]()
    
    override public init() {
        super.init()
    }
    
    public init(_ d: Int..., initValue: Float = 0.0) {
        super.init(repeaing: initValue, count: d.reduce(1, *))
        self.d = d
        setAcci()
    }
    
    public init(_ data: Pointer, d: Int...) {
        super.init(data, d.reduce(1, *), d.reduce(1, *))
        self.d = d
        setAcci()
    }

    init(_ data: [Float], d: [Int]) {
        super.init()
        super.append(contentsOf: data)
        self.d = d
        setAcci()
    }

    @discardableResult
    public func dim(_ d: Int...) -> NNArray {
        precondition(d.reduce(1, *) == count)
        self.d = d
        setAcci()
        return self
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
            return get(getAddr(index))
        }
        set(newValue) {
            set(getAddr(index), newValue)
        }
    }
}
