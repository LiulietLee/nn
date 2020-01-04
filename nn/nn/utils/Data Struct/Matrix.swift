//
//  Matrix.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Matrix: NSObject {
    @objc dynamic public var _data = NNArray()
    
    var row: Int { return _data.d[0] }
    var col: Int { return _data.d[1] }
    
    public init(row: Int = 1, col: Int = 1) {
        _data = NNArray(row, col)
    }
    
    init(_ data: NNArray) {
        self._data = data
    }
    
    public func copy() -> Matrix {
        return Matrix(_data.copy())
    }
    
    @discardableResult
    public func rand() -> Matrix {
        for i in 0..<row {
            for j in 0..<col {
                _data[i, j] = Float.random(in: -1.0..<1.0)
            }
        }
        return self
    }
    
    @discardableResult
    public func ones() -> Matrix {
        for i in 0..<row {
            for j in 0..<col {
                _data[i, j] = 1.0
            }
        }
        return self
    }
    
    public subscript(index1: Int, index2: Int) -> Float {
        get {
            return _data[index1, index2]
        }
        set(newValue) {
            _data[index1, index2] = newValue
        }
    }
}

public func * (lhs: Matrix, rhs: Matrix) -> Matrix {
    let row = lhs.row, col = rhs.col, len = lhs.col
    let out = Matrix(row: row, col: col)
    for i in 0..<row {
        for j in 0..<col {
            for k in 0..<len {
                out[i, j] += lhs[i, k] * rhs[k, j]
            }
        }
    }
    return out
}

public func *= (lhs: inout Matrix, rhs: Matrix) {
    lhs = lhs * rhs
}

public func + (lhs: Matrix, rhs: Matrix) -> Matrix {
    let row = lhs.row
    let col = lhs.col
    let out = Matrix(row: row, col: col)
    for i in 0..<row {
        for j in 0..<col {
            out[i, j] = lhs[i, j] + rhs[i, j]
        }
    }
    return out
}

public func += (lhs: inout Matrix, rhs: Matrix) {
    let row = lhs.row
    let col = lhs.col
    for i in 0..<row {
        for j in 0..<col {
            lhs[i, j] += rhs[i, j]
        }
    }
}

public func * (lhs: Matrix, rhs: NNArray) -> NNArray {
    let row = lhs.row
    let col = lhs.col
    let output = NNArray(row, initValue: 0.0)
    for i in 0..<row {
        for j in 0..<col {
            output[i] += lhs[i, j] * rhs[j]
        }
    }
    return output
}

public func + (lhs: NNArray, rhs: NNArray) -> NNArray {
    let output = NNArray(lhs.count, initValue: 0.0)
    for i in 0..<lhs.count {
        output[i] = lhs[i] + rhs[i]
    }
    return output
}

public func += (lhs: inout NNArray, rhs: NNArray) {
    for i in 0..<lhs.count {
        lhs[i] += rhs[i]
    }
}
