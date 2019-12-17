//
//  ReLU.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class ReLU: Layer {
    public var score = NNArray()
    
    public func backward(_ node: NNArray, derivative: NNArray, rate: Float = 0.1) -> NNArray {
        return NNArray(zip(node, derivative).map {
            return $0.0 >= 0.0 ? $0.1 : 0.0
        }, d: node.d)
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        score = input
        for i in 0..<input.count {
            score[i] = max(0, input[i])
        }
        return score
    }
}
