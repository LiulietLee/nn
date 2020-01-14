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
    
    @objc dynamic var param: NNArray
    @objc dynamic var bias: NNArray

    var interScore = NNArray()

    var vparam = NNArray()
    var vbias = NNArray()
    
    var dparam = NNArray()
    var dbias = NNArray()
    
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, relu: Bool = true) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        needBias = bias
        self.relu = relu
        
        self.bias = NNArray(outFeatures)
        param = NNArray(outFeatures, inFeatures)
        param.normalRandn(n: outFeatures + inFeatures)
    }
    
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            batchSize = input.d[0]
            score = NNArray(batchSize, outFeatures)
            interScore = NNArray(batchSize, outFeatures)
            vparam = NNArray(batchSize, outFeatures, inFeatures)
            vbias = NNArray(batchSize, outFeatures)
            dparam = NNArray(batchSize, outFeatures, inFeatures)
            dbias = NNArray(batchSize, outFeatures)
        }
        
        let inputd = input.d
        input.dim([batchSize, inFeatures])

        if Core.device != nil {
            forwardWithMetal(input)
            return score
        }
        
        interScore.data.zero()
        
        for batch in 0..<batchSize {
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    interScore[batch, i] += param[i, j] * input[batch, j]
                }
            }

        
            for i in 0..<outFeatures {
                score[batch, i] = interScore[batch, i]
                if relu && interScore[batch, i] < 0.0 {
                    score[batch, i] *= 0.001
                }
            }
            
            if needBias {
                for i in 0..<outFeatures {
                    score[batch, i] += bias[i]
                }
            }
        }
        
        input.dim(inputd)
        return score
    }
    
    public override func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        if Core.device != nil {
            return backwardWithMetal(input, delta)
        }
        
        let inputd = input.d
        input.dim([batchSize, inFeatures])
        let da = NNArray(input.count)
        da.dim(input.d)
        
        for batch in 0..<batchSize {
            if needBias {
                // bias.count = outFeature
                for i in 0..<outFeatures {
                    dbias[batch, i] += delta[batch, i]
                }
            }
            
            // score.count = outFeature, input.count = inFeature
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    if relu {
                        da[batch, j] += (interScore[batch, i] >= 0.0 ? 1.0 : 0.001)
                            * delta[batch, i] * param[i, j]
                    } else {
                        da[batch, j] += delta[batch, i] * param[i, j]
                    }
                }
            }
            
            // param.row = outFeature, param.col = inFeature
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    if relu {
                        dparam[batch, i, j] += (interScore[batch, i] >= 0.0 ? 1.0 : 0.001) * delta[batch, i] * input[batch, j]
                    } else {
                        dparam[batch, i, j] += delta[batch, i] * input[batch, j]
                    }
                }
            }
        }
        
        input.dim(inputd)
        return da
    }
    
    public override func step(lr: Float, momentum: Float) {
        if Core.device != nil {
            if needBias {
                stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dbias, v: vbias, p: bias)
            }
            stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dparam, v: vparam, p: param)
            return
        }
        
        for batch in 0..<batchSize {
            if needBias {
                for i in 0..<outFeatures {
                    vbias[batch, i] = momentum * vbias[batch, i] + dbias[batch, i]
                    bias[i] -= lr * vbias[batch, i]
                }
            }
            
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    vparam[batch, i, j] = momentum * vparam[batch, i, j] + dparam[batch, i, j]
                    param[i, j] -= lr * vparam[batch, i, j]
                }
            }
        }
    }
    
    public override func zeroGrad() {
        dparam.data.zero()
        dbias.data.zero()
    }
}
