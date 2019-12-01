//
//  Loss.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

class Loss {
    public static func mod2(score: [Float], label: [Float], delta: Float = 1.0) -> Float {
        var loss: Float = 0.0
        for i in 0 ..< score.count {
            loss += (score[i] - label[i]) * (score[i] - label[i])
        }
        return loss
    }
}
