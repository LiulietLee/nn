//
//  d2.swift
//  nn
//
//  Created by Liuliet.Lee on 14/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

extension Loss {
    public class d2: AbstractLoss {
        public static func loss(score: NNArray, label: NNArray) -> Float {
            var loss: Float = 0.0
            for i in 0..<score.count {
                loss += (score[i] - label[i]) * (score[i] - label[i])
            }
            return loss
        }
        
        public static func delta(score: NNArray, label: NNArray) -> NNArray {
            return NNArray(zip(label, score).map { return -2.0 * ($0.0 - $0.1) }, d: score.d)
        }
    }
}
