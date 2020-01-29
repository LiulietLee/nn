//
//  Conv.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

/**
 2D Convolution Layer
 */
public class Conv: BaseLayer {

    @objc dynamic var core = NNArray()
    @objc dynamic var bias: NNArray
    
    @objc dynamic var vcore = NNArray()
    @objc dynamic var vbias = NNArray()
    
    @objc dynamic var mcore = NNArray()
    @objc dynamic var mbias = NNArray()
    
    var dcore = NNArray()
    var dbias = NNArray()
    
    @objc dynamic var row = 0
    @objc dynamic var col = 0
    @objc dynamic var depth = 0

    var needBias = true
    var width = 0
    var height = 0
    var count = 0
    var step = 0
    var padding = 0
    
    /**
     Unlike PyTorch, you don't need to specify the input channel, which will be set automatically while the first time of `forward()` function execution.
     
     - parameter width: Size of the convolving kernel.
     - parameter height: Ignore this.
     - parameter count: Number of channels produced by the convolution.
     - parameter step: Stride for convolution.
     - parameter padding: Zero-padding added to both sides of the input.
     - parameter bias: Whether to use bias.
     */
    public init(_ width: Int, _ height: Int = -1, count: Int = 1, step: Int = 1, padding: Int = 0, bias: Bool = true) {
        self.width = width
        self.height = height < 0 ? width : height
        self.count = count
        self.step = step
        self.padding = padding
        needBias = bias
        self.bias = NNArray(count)
    }

    private func inBound(_ x: Int, _ y: Int, _ row: Int, _ col: Int) -> Bool {
        return 0 <= x && x < row && 0 <= y && y < col
    }
    
    /**
     Call this function to do prediction.
     
     - parameter input: Shape: [batch_size, input_channel, height, width]
     - returns: Shape: [batch_size, count, output_height, output_width]
     */
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            precondition(
                (input.d[2] - width + padding * 2) % step == 0 &&
                (input.d[3] - height + padding * 2) % step == 0
            )
            batchSize = input.d[0]
            depth = input.d[1]
            row = (input.d[2] - width + padding * 2) / step + 1
            col = (input.d[3] - height + padding * 2) / step + 1
            
            if core.d == [] {
                core = NNArray(count, depth, width, height)
                core.normalRandn(n: input.count + score.count)
                vcore = NNArray(batchSize, count, depth, width, height)
                vbias = NNArray(batchSize, count)
                mcore = NNArray(batchSize, count, depth, width, height)
                mbias = NNArray(batchSize, count)
            }
            
            dcore = NNArray(batchSize, count, depth, width, height)
            dbias = NNArray(batchSize, count)
            
            score = NNArray(batchSize, count, row, col)
        }
        score.data.zero()
        
        if Core.device != nil {
            forwardWithMetal(input)
            return score
        }
        
        for batch in 0..<batchSize {
            for c in 0..<count {
                for i in 0..<row {
                    for j in 0..<col {
                        for x in 0..<width {
                            for y in 0..<height {
                                for z in 0..<depth {
                                    let rx = i * step + x - padding, ry = j * step + y - padding
                                    if inBound(rx, ry, input.d[2], input.d[3]) {
                                        score[batch, c, i, j] += input[batch, z, rx, ry] * core[c, z, x, y]
                                    }
                                }
                            }
                        }
                        if needBias {
                            score[batch, c, i, j] += bias[c]
                        }
                    }
                }
            }
        }
        
        return score
    }

    public override func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        delta.dim(score.d)

        if Core.device != nil {
            return backwardWithMetal(input, delta)
        }
        
        let da = NNArray(input.count)
        da.dim(input.d)
        
        for batch in 0..<batchSize {
            if needBias {
                for c in 0..<count {
                    var sum: Float = 0.0
                    for i in 0..<row {
                        for j in 0..<col {
                            sum += delta[batch, c, i, j]
                        }
                    }
                    dbias[batch, c] += sum
                }
            }
            
            for rx in 0..<input.d[2] {
                for ry in 0..<input.d[3] {
                    for z in 0..<input.d[1] {
                        for c in 0..<count {
                            for x in 0..<width {
                                for y in 0..<height {
                                    if (rx + padding - x) % step == 0,
                                        (ry + padding - y) % step == 0 {
                                        let i = (rx + padding - x) / step
                                        let j = (ry + padding - y) / step
                                        if inBound(i, j, row, col) {
                                            da[batch, z, rx, ry] += core[c, z, x, y] * delta[batch, c, i, j]
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            for c in 0..<count {
                for x in 0..<width {
                    for y in 0..<height {
                        for z in 0..<depth {
                            for i in 0..<row {
                                for j in 0..<col {
                                    let rx = i * step + x - padding, ry = j * step + y - padding
                                    if inBound(rx, ry, input.d[2], input.d[3]) {
                                        dcore[batch, c, z, x, y] += input[batch, z, rx, ry] * delta[batch, c, i, j]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return da
    }
    
    public override func step(lr: Float, momentum: Float) {
        if Core.device != nil {
            if needBias {
                stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dbias, m: mbias, v: vbias, p: bias)
            }
            stepWithMetal(batch: batchSize, lr: lr, momentum: momentum, d: dcore, m: mcore, v: vcore, p: core)
            return
        }
        
        for batch in 0..<batchSize {
            if needBias {
                for i in 0..<bias.count {                    
                    mbias[batch, i] = 0.9 * mbias[batch, i] + (1.0 - 0.9) * dbias[batch, i]
                    vbias[batch, i] = momentum * vbias[batch, i] + (1.0 - momentum) * dbias[batch, i] * dbias[batch, i]
                    bias[i] -= lr * mbias[batch, i] / (sqrt(vbias[batch, i]) + 1e-8)
                }
            }
            
            for i in 0..<core.count {
                let idx = batch * core.count + i
                mcore[idx] = 0.9 * mcore[idx] + (1.0 - 0.9) * dcore[idx]
                vcore[idx] = momentum * vcore[idx] + (1.0 - momentum) * dcore[idx] * dcore[idx]
                core[i] -= lr * mcore[idx] / (sqrt(vcore[idx]) + 1e-8)
            }
        }
    }
    
    public override func zeroGrad() {
        dcore.data.zero()
        dbias.data.zero()
    }
}
