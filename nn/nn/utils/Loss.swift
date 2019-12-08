//
//  Loss.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol AbstractLoss {
    static func loss(score: [Float], label: [Float]) -> Float
    static func derivative(score: [Float], label: [Float]) -> [Float]
}

public class Loss {
    public class mod2: AbstractLoss {
        public static func loss(score: [Float], label: [Float]) -> Float {
            var loss: Float = 0.0
            for i in 0 ..< score.count {
                loss += (score[i] - label[i]) * (score[i] - label[i])
            }
            return loss
        }
        
        public static func derivative(score: [Float], label: [Float]) -> [Float] {
            return zip(label, score).map { return -2.0 * ($0.0 - $0.1) }
        }
    }
}
