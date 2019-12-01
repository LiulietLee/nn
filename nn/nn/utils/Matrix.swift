//
//  Matrix.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Matrix {
    private var _data: [[Float]] = []
    
    private(set) var row: Int = 0
    private(set) var col: Int = 0
    
    public init(row: Int = 1, col: Int = 1) {
        self.row = row
        self.col = col
        _data = Array.init(
            repeating: Array.init(repeating: 0.0, count: col),
            count: row
        )
    }
    
    private init(_ data: [[Float]]) {
        self.row = data.count
        self.col = data[0].count
        self._data = data
    }
    
    public func copy() -> Matrix {
        return Matrix(_data)
    }
    
    public func rand() {
        for i in 0 ..< row {
            for j in 0 ..< col {
                _data[i][j] = Float.random(in: -1.0..<1.0)
            }
        }
    }
    
    public func ones() {
        for i in 0 ..< row {
            for j in 0 ..< col {
                _data[i][j] = 1.0
            }
        }
    }
    
    public subscript(index1: Int, index2: Int) -> Float {
        get {
            return _data[index1][index2]
        }
        set(newValue) {
            _data[index1][index2] = newValue
        }
    }
}

public func * (lhs: Matrix, rhs: Matrix) -> Matrix {
    let row = lhs.row, col = rhs.col, len = lhs.col
    let out = Matrix(row: row, col: col)
    for i in 0 ..< row {
        for j in 0 ..< col {
            for k in 0 ..< len {
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
    for i in 0 ..< row {
        for j in 0 ..< col {
            out[i, j] = lhs[i, j] + rhs[i, j]
        }
    }
    return out
}

public func += (lhs: inout Matrix, rhs: Matrix) {
    let row = lhs.row
    let col = lhs.col
    for i in 0 ..< row {
        for j in 0 ..< col {
            lhs[i, j] += rhs[i, j]
        }
    }
}
