//
//  Layer.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

/**
 Fully Connected Layer.
 
 `Y = WX + b`
 */
public class Dense: BaseLayer {
    var inFeatures = 0
    var outFeatures = 0
    var needBias = false
    var relu = true
    
    @objc dynamic var param: NNArray
    @objc dynamic var bias: NNArray

    var interScore = NNArray()

    @objc dynamic var vparam = NNArray()
    @objc dynamic var vbias = NNArray()

    @objc dynamic var mparam = NNArray()
    @objc dynamic var mbias = NNArray()

    var dparam = NNArray()
    var dbias = NNArray()
    
    /**
     - parameter inFeatures: Size of each input sample.
     - parameter outFeatures: Size of each output sample.
     - parameter bias: Whether to use bias.
     - parameter relu: Whether to use relu.
     */
    public init(inFeatures: Int, outFeatures: Int, bias: Bool = true, relu: Bool = true) {
        self.inFeatures = inFeatures
        self.outFeatures = outFeatures
        needBias = bias
        self.relu = relu
        
        self.bias = NNArray(outFeatures)
        param = NNArray(outFeatures, inFeatures)
        param.normalRandn(n: outFeatures + inFeatures)
    }
    
    /**
     Call this function to do prediction.
     
     - parameter input: Shape: [batch_size, inFeatures]
     - returns: Shape: [batch_size, outFeatures]
     */
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            batchSize = input.d[0]
            score = NNArray(batchSize, outFeatures)
            interScore = NNArray(batchSize, outFeatures)
            vparam = NNArray(batchSize, outFeatures, inFeatures)
            vbias = NNArray(batchSize, outFeatures)
            mparam = NNArray(batchSize, outFeatures, inFeatures)
            mbias = NNArray(batchSize, outFeatures)
            dparam = NNArray(batchSize, outFeatures, inFeatures)
            dbias = NNArray(batchSize, outFeatures)
        }
        
        let inputd = input.d
        input.dim([batchSize, inFeatures])
        interScore.data.zero()

        forwardCompute(input: input, inter: interScore, output: score, batch: batchSize)
        
        input.dim(inputd)
        return score
    }
    
    public override func predict(_ input: NNArray) -> NNArray {
        let input: NNArray = input.copy()
        let score = NNArray(input.d[0], outFeatures)
        let inter = NNArray(input.d[0], outFeatures)

        let inputd = input.d
        input.dim([input.d[0], inFeatures])

        forwardCompute(input: input, inter: inter, output: score, batch: input.d[0])
        
        input.dim(inputd)
        return score
    }
    
    func forwardCompute(input: NNArray, inter: NNArray, output: NNArray, batch: Int) {
        if Core.device != nil {
            forwardWithMetal(batch, input, inter, output)
            return
        }
                
        for batch in 0..<batchSize {
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    inter[batch, i] += param[i, j] * input[batch, j]
                }
            }

        
            for i in 0..<outFeatures {
                output[batch, i] = inter[batch, i]
                if relu && inter[batch, i] < 0.0 {
                    output[batch, i] *= 0.001
                }
            }
            
            if needBias {
                for i in 0..<outFeatures {
                    output[batch, i] += bias[i]
                }
            }
        }
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
                stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dbias, m: mbias, v: vbias, p: bias)
            }
            stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dparam, m: mparam, v: vparam, p: param)
            return
        }

        for batch in 0..<batchSize {
            if needBias {
                for i in 0..<outFeatures {
                    mbias[batch, i] = 0.9 * mbias[batch, i] + (1 - 0.9) * dbias[batch, i]
                    vbias[batch, i] = momentum * vbias[batch, i] + (1 - momentum) * dbias[batch, i] * dbias[batch, i]
                    bias[i] -= lr * mbias[batch, i] / (sqrt(vbias[batch, i]) + 1e-8)
                }
            }
            
            for i in 0..<outFeatures {
                for j in 0..<inFeatures {
                    mparam[batch, i, j] = 0.9 * mparam[batch, i, j] + (1 - 0.9) * dparam[batch, i, j]
                    vparam[batch, i, j] = momentum * vparam[batch, i, j] + (1 - momentum) * dparam[batch, i, j] * dparam[batch, i, j]
                    param[i, j] -= lr * mparam[batch, i, j] / (sqrt(vparam[batch, i, j]) + 1e-8)
                }
            }
        }
    }
    
    public override func zeroGrad() {
        dparam.data.zero()
        dbias.data.zero()
    }
}
