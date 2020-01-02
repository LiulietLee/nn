//
//  ReLU.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class ReLU: BaseLayer {
    
    public override func forward(_ input: NNArray) -> NNArray {
        score = input
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        for i in 0..<input.count {
            score[i] = max(input[i] * 0.001, input[i])
        }
        return score
    }
    
    public override func backward(_ input: NNArray, delta: NNArray, rate: Float = 0.1) -> NNArray {
        if Core.device != nil {
            return backwardWithMetal(input, delta)
        } else {
            return NNArray(zip(input, delta).map {
                return $0.0 >= 0.0 ? $0.1 : 0.001 * $0.1
            }, d: input.d)
        }
    }
}
