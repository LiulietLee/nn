//
//  Layer.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Dense: BaseLayer {
    var inFeatures = 0
    var outFeatures = 0
    var needBias = false
    var relu = true
    @objc dynamic var bias = NNArray()
    
    public var interScore = NNArray()
    @objc dynamic var param: Matrix
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, relu: Bool = true) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        needBias = bias
        self.relu = relu
        self.bias = NNArray(outFeatures, initValue: needBias ? 0.0001 : Float(0.0))
        param = Matrix(row: outFeatures, col: inFeatures)
        interScore = NNArray(outFeatures)

        super.init()
        score = NNArray(outFeatures)
    }
    
    public override func forward(_ input: NNArray) -> NNArray {
        if Core.device != nil {
            forwardWithMetal(input)
        } else {
            if relu {
                interScore = param * input + bias
                score = NNArray(interScore.map { return $0 > 0.0 ? $0 : 0.001 * $0 }, d: [outFeatures])
            } else {
                score = param * input + bias
            }
        }
        return score
    }
    
    public override func backward(_ input: NNArray, delta: NNArray, rate: Float = 0.1) -> NNArray {
        if Core.device != nil {
            return backwardWithMetal(input, delta, rate)
        }
        
        let da = NNArray(input.count, initValue: 0.0)
        
        if needBias {
            // bias.count = outFeature
            for i in 0..<outFeatures {
                bias[i] -= delta[i] * rate
            }
        }
        
        // score.count = outFeature, input.count = inFeature
        for i in 0..<outFeatures {
            for j in 0..<inFeatures {
                if relu {
                    da[j] += (interScore[i] >= 0.0 ? 1.0 : 0.001)
                        * delta[i] * param[i, j] * rate
                } else {
                    da[j] += delta[i] * param[i, j] * rate
                }
            }
        }
        
        // param.row = outFeature, param.col = inFeature
        for i in 0..<outFeatures {
            for j in 0..<inFeatures {
                if relu {
                    param[i, j] -= (interScore[i] >= 0.0 ? 1.0 : 0.001)
                        * delta[i] * input[j] * rate
                } else {
                    param[i, j] -= delta[i] * input[j] * rate
                }
            }
        }
        
        return da
    }
}
