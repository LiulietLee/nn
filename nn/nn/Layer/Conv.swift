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
    private var symbols = NNArray()
    private var bias = NNArray()
    
    var needBias = true
    var width = 0
    var height = 0
    var depth = 0
    var count = 0
    var step = 0
    var padding = 0
    var row = 0
    var col = 0
    
    public init(_ width: Int, _ height: Int, count: Int = 1, step: Int = 1, padding: Int = 0, bias: Bool = true) {
        self.width = width
        self.height = height
        self.count = count
        self.step = step
        self.padding = padding
        needBias = bias
    }
    
    func setDepth(_ depth: Int) {
        self.depth = depth
        symbols = NNArray(width, height, depth, count, initValue: 0.01)
    }
    
    private func inBound(_ x: Int, _ y: Int, _ arr: NNArray) -> Bool {
        return 0 <= x && x < arr.d[0] && 0 <= y && y < arr.d[1]
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        if row == 0 {
            setDepth(input.d[2])
            row = input.d[0] - width + 1 + padding * 2
            col = input.d[1] - height + 1 + padding * 2
            if needBias {
                bias = NNArray(count, initValue: 0.02)
            }
        }
        score = NNArray(row, col, count)
        
        for c in 0..<count {
            for i in 0..<row {
                for j in 0..<col {
                    for x in 0..<width {
                        for y in 0..<height {
                            for z in 0..<depth {
                                let rx = i + x - padding, ry = j + y - padding
                                if inBound(rx, ry, input) {
                                    score[i, j, c] += input[rx, ry, z] * symbols[x, y, z, c]
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
        
        return score.copy()
    }
    
    public func backward(_ node: NNArray, derivative: NNArray, rate: Float = 0.1) -> NNArray {
        let da = NNArray(node.count, initValue: 0.0)
        da.dim(node.d)
        derivative.dim(score.d)
        
        if needBias {
            for c in 0..<count {
                var sum: Float = 0.0
                for i in 0..<row {
                    for j in 0..<col {
                        sum += derivative[i, j, c]
                    }
                }
                bias[c] -= sum * rate
            }
        }
        
        for c in 0..<count {
            for x in 0..<width {
                for y in 0..<height {
                    for z in 0..<depth {
                        for i in 0..<row {
                            for j in 0..<col {
                                let rx = i + x - padding, ry = j + y - padding
                                if inBound(rx, ry, node) {
                                    da[rx, ry, z] += symbols[x, y, z, c] * derivative[i, j, c] * rate
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
                                let rx = i + x - padding, ry = j + y - padding
                                if inBound(rx, ry, node) {
                                    symbols[x, y, z, c] -= node[rx, ry, z] * derivative[i, j, c] * rate
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
