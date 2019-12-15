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
    private var bias = [Float]()
    public var score = [Float]()
    public var interScore = [Float]()
    private var param: Matrix
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, relu: Bool = true) {
        self.inFeatures = outFeatures
        self.outFeatures = outFeatures
        needBias = bias
        self.relu = relu
        
        self.bias = Array.init(
            repeating: needBias ? 0.1 : Float(0.0),
            count: outFeatures
        )
        
        param = Matrix(row: outFeatures, col: inFeatures).rand()
    }
    
    public func forward(_ input: [Float]) -> [Float] {
        if relu {
            interScore = param * input + bias
            score = interScore.map { return $0 > 0.0 ? $0 : 0.0 }
        } else {
            score = param * input + bias
        }
        
        return score
    }
    
    public func backward(_ node: [Float], derivative: [Float], rate: Float = 0.1) -> [Float] {
        var da = Array.init(repeating: Float(0.0), count: node.count)
        
        if needBias {
            bias = zip(bias, derivative).map { return $0.0 - $0.1 * rate }
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
