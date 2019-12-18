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
    var symbols = NNArray()
    var width = 0
    var height = 0
    var depth = 0
    var count = 0
    var step = 0
    
    var row = 0
    var col = 0
    
    public init(_ width: Int, _ height: Int, count: Int = 1, step: Int = 1) {
        self.width = width
        self.height = height
        self.count = count
        self.step = step
    }
    
    func setDepth(_ depth: Int) {
        self.depth = depth
        symbols = NNArray(width, height, depth, count, initValue: 0.1)
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        if row == 0 {
            setDepth(input.d[2])
            row = input.d[0] - width + 1
            col = input.d[1] - height + 1
        }
        score = NNArray(row, col, count)
        
        for c in 0..<count {
            for i in 0..<row {
                for j in 0..<col {
                    for x in 0..<width {
                        for y in 0..<height {
                            for z in 0..<depth {
                                score[i, j, c] += input[i + x, j + y, z] * symbols[x, y, z, c]
                            }
                        }
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
        
        for c in 0..<count {
            for x in 0..<width {
                for y in 0..<height {
                    for z in 0..<depth {
                        for i in 0..<row {
                            for j in 0..<col {
                                da[x + i, y + j, z] += symbols[x, y, z, c] * derivative[i, j, c] * rate
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
                                symbols[x, y, z, c] -= node[x + i, y + j, z] * derivative[i, j, c] * rate
                            }
                        }
                    }
                }
            }
        }
        
        return da
    }
    
}
