//
//  MaxPooling.swift
//  nn
//
//  Created by Liuliet.Lee on 19/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class MaxPooling: BaseLayer {
        
    public struct SwitchMapper {
        var oi: Int32, oj: Int32, ix: Int32, iy: Int32, k: Int32
    }
    
    public var switches = LLVector<SwitchMapper>()
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
    
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            precondition(
                (input.d[2] - width) % step == 0 &&
                (input.d[3] - width) % step == 0
            )
            batchSize = input.d[0]
            row = (input.d[2] - width) / step + 1
            col = (input.d[3] - width) / step + 1
            switches = LLVector<SwitchMapper>(
                capacity: batchSize * input.d[1] * row * col
            )
            switches.length = batchSize * input.d[1] * row * col
            score = NNArray(batchSize, input.d[1], row, col)
        }
        score.data.zero()
        
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        
        for batch in 0..<batchSize {
            for k in 0..<input.d[1] {
                for i in 0..<row {
                    let ri = i * step
                    for j in 0..<col {
                        let rj = j * step
                        var maxPostion = (ri, rj)
                        var maxValue = input[batch, k, ri, rj]
                        for x in 0..<width {
                            for y in 0..<height {
                                let rx = ri + x, ry = rj + y
                                if maxValue < input[batch, k, rx, ry] {
                                    maxValue = input[batch, k, rx, ry]
                                    maxPostion = (rx, ry)
                                }
                            }
                        }
                        switches[batch * input.d[1] * row * col +
                            k * row * col + i * col + j] = SwitchMapper(
                            oi: Int32(i), oj: Int32(j), ix: Int32(maxPostion.0), iy: Int32(maxPostion.1), k: Int32(k)
                        )
                        score[batch, k, i, j] = maxValue
                    }
                }
            }
        }
        
        return score
    }
    
    public override func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        let da = NNArray(input.count, initValue: 0.0).dim(input.d)
        delta.dim(score.d)
        
        if Core.device != nil {
            return backwardWithMetal(input, delta)
        }
        
        for choose in switches {
            da[Int(choose.ix), Int(choose.iy), Int(choose.k)] =
                delta[Int(choose.oi), Int(choose.oj), Int(choose.k)]
        }
        
        return da
    }
}
