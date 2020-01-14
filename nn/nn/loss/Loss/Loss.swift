//
//  Loss.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol AbstractLoss {
    static func loss(score: NNArray, label: NNArray) -> Float
    static func delta(score: NNArray, label: NNArray) -> NNArray
}

public class Loss {
}
