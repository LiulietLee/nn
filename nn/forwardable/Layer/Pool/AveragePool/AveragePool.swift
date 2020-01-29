//
//  AveragePool.swift
//  nn
//
//  Created by Liuliet.Lee on 22/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

/**
 2D Average Pooling Layer (underbuilding yet).
 
 The step and the size have to be the same value because of the uncompleted backward function. So it's recommended to only specify the width value.
 */
public class AveragePool: BasePool {
    
    /**
     - parameter width: Size of the convolving kernel.
     */
    public init(_ width: Int) {
        super.init(width, step: width)
    }
    
    /**
     Call this function to do prediction.
     
     - parameter input: Shape: [batch_size, input_channel, height, width]
     - returns: Shape: [batch_size, input_channel, output_height, output_width]
     */
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            precondition(
                (input.d[2] - width) % step == 0 &&
                (input.d[3] - height) % step == 0
            )
            batchSize = input.d[0]
            row = (input.d[2] - width) / step + 1
            col = (input.d[3] - width) / step + 1
            score = NNArray(batchSize, input.d[1], row, col)
        }
        score.data.zero()
        
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        
        for batch in 0..<batchSize {
            for k in 0..<input.d[1] {
                for 	i in 0..<row {
                    let ri = i * step
                    for j in 0..<col {
                        let rj = j * step
                        var sum: Float = 0.0
                        for x in 0..<width {
                            for y in 0..<height {
                                let rx = ri + x, ry = rj + y
                                sum += input[batch, k, rx, ry]
                            }
                        }

                        score[batch, k, i, j] = sum / Float(width * height)
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
            return backwardWithMetal(da, input, delta)
        }
        
        for batch in 0..<batchSize {
            for k in 0..<input.d[1] {
                for i in 0..<row {
                    let ri = i * step
                    for j in 0..<col {
                        let rj = j * step
                        for x in 0..<width {
                            for y in 0..<height {
                                let rx = ri + x, ry = rj + y
                                da[batch, k, rx, ry] = delta[batch, k, i, j] / Float(width * height)
                            }
                        }
                    }
                }
            }
        }
        
        return da
    }
}
