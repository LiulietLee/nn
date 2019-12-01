//
//  ReLU.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class ReLU: Layer {
    public var score: [Float] = []
    
    public func backward(_ node: [Float], derivative: [Float], lr: Float = 0.1) -> [Float] {
        return zip(node, derivative).map {
            return $0.0 >= 0.0 ? $0.1 : 0.0
        }
    }
    
    public func forward(_ input: [Float]) -> [Float] {
        score = input
        for i in 0 ..< input.count {
            score[i] = max(0, input[i])
        }
        return score
    }
}
