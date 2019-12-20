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
    private(set) var relu = true
    private var bias = NNArray()
    public var score = NNArray()
    public var interScore = NNArray()
    private var param: Matrix
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, relu: Bool = true) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        needBias = bias
        self.relu = relu
        self.bias = NNArray(outFeatures, initValue: needBias ? 0.01 : Float(0.0))
        param = Matrix(row: outFeatures, col: inFeatures)
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        if relu {
            interScore = param * input + bias
            score = NNArray(interScore.map { return $0 > 0.0 ? $0 : 0.0 }, d: [outFeatures])
        } else {
            score = param * input + bias
        }
        
        return score
    }
    
    public func backward(_ node: NNArray, derivative: NNArray, rate: Float = 0.1) -> NNArray {
        let da = NNArray(node.count, initValue: 0.0)
        
        if needBias {
            bias = NNArray(zip(bias, derivative).map { return $0.0 - $0.1 * rate }, d: [outFeatures])
        }
        
        for i in 0..<score.count {
            for j in 0..<node.count {
                if relu {
                    da[j] += (interScore[i] >= 0.0 ? 1.0 : 0.0)
                        * derivative[i] * param[i, j] * rate
                } else {
                    da[j] += derivative[i] * param[i, j] * rate
                }
            }
        }
        
        for j in 0..<param.row {
            for i in 0..<param.col {
                if relu {
                    param[j, i] -= (interScore[j] >= 0.0 ? 1.0 : 0.0)
                        * derivative[j] * node[i] * rate
                } else {
                    param[j, i] -= derivative[j] * node[i] * rate
                }
            }
        }
        
        return da
    }
}
