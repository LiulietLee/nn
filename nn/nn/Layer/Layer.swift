//
//  Layer.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Layer {
    var score: [Float] { get set }
    func forward(_ input: [Float]) -> [Float]
    func backward(_ node: [Float], derivative: [Float], rate: Float) -> [Float]
}
