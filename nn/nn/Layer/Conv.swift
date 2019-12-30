//
//  Conv.swift
//  nn
//
//  Created by Liuliet.Lee on 8/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Conv: Layer {
    public var score = NNArray()
    var convCore = NNArray()
    var bias = NNArray()
    
    var needBias = true
    var width = 0
    var height = 0
    var depth = 0
    var count = 0
    var step = 0
    var padding = 0
    var row = 0
    var col = 0
    
    public init(_ width: Int, _ height: Int = -1, count: Int = 1, step: Int = 1, padding: Int = 0, bias: Bool = true) {
        self.width = width
        self.height = height < 0 ? width : height
        self.count = count
        self.step = step
        self.padding = padding
        needBias = bias
        self.bias = NNArray(count, initValue: 0.0)
    }
    
    func setDepth(_ depth: Int) {
        self.depth = depth
        convCore = NNArray(width, height, depth, count, initValue: 0.0001)
    }
    
    private func inBound(_ x: Int, _ y: Int, _ b: [Int]) -> Bool {
        return 0 <= x && x < b[0] && 0 <= y && y < b[1]
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        if row == 0 {
            precondition(
                (input.d[0] - width + padding * 2) % step == 0 &&
                (input.d[1] - height + padding * 2) % step == 0
            )
            row = (input.d[0] - width + padding * 2) / step + 1
            col = (input.d[1] - height + padding * 2) / step + 1
            setDepth(input.d[2])
        }
        score = NNArray(row, col, count)
        
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        
        for c in 0..<count {
            for i in 0..<row {
                for j in 0..<col {
                    for x in 0..<width {
                        for y in 0..<height {
                            for z in 0..<depth {
                                let rx = i * step + x - padding, ry = j * step + y - padding
                                if inBound(rx, ry, input.d) {
                                    score[i, j, c] += input[rx, ry, z] * convCore[x, y, z, c]
                                }
                            }
                        }
                    }
                    if needBias {
                        score[i, j, c] += bias[c]
                    }
                }
            }
        }
        
        return score
    }
    
    public func backward(_ input: NNArray, delta: NNArray, rate: Float = 0.1) -> NNArray {
        let da = NNArray(input.count, initValue: 0.0)
        da.dim(input.d)
        delta.dim(score.d)
        
        if Core.device != nil {
            return backwardWithMetal(da, input, delta, rate)
        }
        
        if needBias {
            for c in 0..<count {
                var sum: Float = 0.0
                for i in 0..<row {
                    for j in 0..<col {
                        sum += delta[i, j, c]
                    }
                }
                bias[c] -= sum * rate
            }
        }
        
        for rx in 0..<input.d[0] {
            for ry in 0..<input.d[1] {
                for z in 0..<input.d[2] {
                    for c in 0..<count {
                        for x in 0..<width {
                            for y in 0..<height {
                                if (rx + padding - x) % step == 0,
                                    (ry + padding - y) % step == 0 {
                                    let i = (rx + padding - x) / step
                                    let j = (ry + padding - y) / step
                                    if inBound(i, j, [row, col]) {
                                        da[rx, ry, z] += convCore[x, y, z, c] * delta[i, j, c] * rate
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
                                if inBound(rx, ry, input.d) {
                                    convCore[x, y, z, c] -= input[rx, ry, z] * delta[i, j, c] * rate
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return da
    }
    
}
