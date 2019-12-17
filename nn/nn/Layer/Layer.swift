//
//  Layer.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Layer {
    var score: NNArray { get set }
    func forward(_ input: NNArray) -> NNArray
    func backward(_ node: NNArray, derivative: NNArray, rate: Float) -> NNArray
}
