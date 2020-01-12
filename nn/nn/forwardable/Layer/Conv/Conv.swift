//
//  Conv.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Conv: BaseLayer {

    @objc dynamic var convCore = NNArray()
    @objc dynamic var bias: NNArray
    
    var vcore = NNArray()
    var vbias = NNArray()
    
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
    
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            precondition(
                (input.d[1] - width + padding * 2) % step == 0 &&
                (input.d[2] - height + padding * 2) % step == 0
            )
            batchSize = input.d[0]
            depth = input.d[1]
            row = (input.d[2] - width + padding * 2) / step + 1
            col = (input.d[3] - height + padding * 2) / step + 1
            
            convCore = NNArray(count, depth, width, height)
            convCore.normalRandn(n: input.count + score.count)

            vcore = NNArray(batchSize, count, depth, width, height)
            dcore = NNArray(batchSize, count, depth, width, height)
            vbias = NNArray(batchSize, count)
            dbias = NNArray(batchSize, count)
            
            score = NNArray(batchSize, count, row, col)
        }
        score.data.zero()
        
        if Core.device != nil {
            return forwardWithMetal(input)
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
                                        score[batch, c, i, j] += input[batch, z, rx, ry] * convCore[c, z, x, y]
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
        let da = NNArray(input.count, initValue: 0.0)
        da.dim(input.d)
        delta.dim(score.d)
        
        if Core.device != nil {
            return backwardWithMetal(da, input, delta)
        }
        
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
                                            da[batch, z, rx, ry] += convCore[c, z, x, y] * delta[batch, c, i, j]
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
                stepWithMetal(lr: lr, momentum: momentum, d: dbias, v: vbias, p: bias)
            }
            stepWithMetal(lr: lr, momentum: momentum, d: dcore, v: vcore, p: convCore)
            return
        }
        
        if needBias {
            for i in 0..<bias.count {
                vbias[i] = momentum * vbias[i] + dbias[i]
                bias[i] -= lr * vbias[i]
            }
        }
        
        for i in 0..<convCore.count {
            vcore[i] = momentum * vcore[i] + dcore[i]
            convCore[i] -= lr * vcore[i]
        }
    }
    
    public override func zeroGrad() {
        dcore.data.zero()
        dbias.data.zero()
    }
}
