//
//  MaxPooling.swift
//  nn
//
//  Created by Liuliet.Lee on 19/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class MaxPooling: Layer {
    
    public var score = NNArray()
    
    struct SwitchMapper {
        var oi: Int, oj: Int, ix: Int, iy: Int, k: Int        
    }
    
    var switches = LLVector<SwitchMapper>()
    var width = 0
    var height = 0
    var step = 0
    
    var row = 0
    var col = 0
    
    public init(_ width: Int = 2, _ height: Int = 2, step: Int = 2) {
        self.width = width
        self.height = height
        self.step = step
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        if row == 0 {
            precondition(
                (input.d[0] - width) % step == 0 &&
                (input.d[1] - width) % step == 0
            )
            row = (input.d[0] - width) / step + 1
            col = (input.d[1] - width) / step + 1
            switches = LLVector<SwitchMapper>(capacity: row * col * input.d[2])
            switches.length = row * col * input.d[2]
        }
        score = NNArray(row, col, input.d[2], initValue: 0.0)
        
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        
        for k in 0..<input.d[2] {
            for i in 0..<row {
                let ri = i * step
                for j in 0..<col {
                    let rj = j * step
                    var maxPostion = (ri, rj)
                    var maxValue = input[ri, rj, k]
                    for x in 0..<width {
                        for y in 0..<height {
                            let rx = ri + x, ry = rj + y
                            if maxValue < input[rx, ry, k] {
                                maxValue = input[rx, ry, k]
                                maxPostion = (rx, ry)
                            }
                        }
                    }
                    switches[i * col * input.d[2] + j * input.d[2] + k] = SwitchMapper(
                        oi: i, oj: j, ix: maxPostion.0, iy: maxPostion.1, k: k
                    )
                    score[i, j, k] = maxValue
                }
            }
        }
        
        return score
    }
    
    public func backward(_ input: NNArray, delta: NNArray, rate: Float) -> NNArray {
        let da = NNArray(input.count, initValue: 0.0).dim(input.d)
        delta.dim(score.d)
        
        if Core.device != nil {
            return backwardWithMetal(input, delta)
        }
        
        for choose in switches {
            da[choose.ix, choose.iy, choose.k] = delta[choose.oi, choose.oj, choose.k]
        }
        
        return da
    }
}
