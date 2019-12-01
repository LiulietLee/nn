//
//  Layer.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Dense: Layer {
    private(set) var inFeatures = 0
    private(set) var outFeatures = 0
    private(set) var needBias = false
    
    private var bias = [Float]()
    public var score = [Float]()
    private var param: Matrix
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true) {
        self.inFeatures = outFeatures
        self.outFeatures = outFeatures
        needBias = bias
        
        self.bias = Array.init(
            repeating: needBias ? Float.random(in: -1.0 ..< 1.0) : Float(0.0),
            count: outFeatures
        )
        
        param = Matrix(row: outFeatures, col: inFeatures)
        param.rand()
    }
    
    public func forward(_ input: [Float]) -> [Float] {
        score = param * input + bias
        return score
    }
    
    public func backward(_ node: [Float], derivative: [Float], lr: Float = 0.1) -> [Float] {
        var da = Array.init(repeating: Float(0.0), count: node.count)
        
        if needBias {
            bias = zip(bias, derivative).map { return $0.0 - $0.1 * lr }
        }
        
        for i in 0..<score.count {
            for j in 0..<node.count {
                da[j] += derivative[i] * param[i, j] * 0.1
            }
        }
        
        for j in 0..<param.row {
            for i in 0..<param.col {
                param[j, i] -= derivative[j] * node[i] * 0.1
            }
        }
        
        return da
    }
}

public func * (lhs: Matrix, rhs: [Float]) -> [Float] {
    let row = lhs.row
    let col = lhs.col
    var output = Array.init(repeating: Float(0.0), count: row)
    for i in 0 ..< row {
        for j in 0 ..< col {
            output[i] += lhs[i, j] * rhs[j]
        }
    }
    return output
}

public func + (lhs: [Float], rhs: [Float]) -> [Float] {
    var output = Array.init(repeating: Float(0.0), count: lhs.count)
    for i in 0 ..< lhs.count {
        output[i] = lhs[i] + rhs[i]
    }
    return output
}

public func += (lhs: inout [Float], rhs: [Float]) {
    for i in 0 ..< lhs.count {
        lhs[i] += rhs[i]
    }
}
