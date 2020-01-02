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
    func backward(_ input: NNArray, delta: NNArray, rate: Float) -> NNArray
    func save(to file: UnsafeMutablePointer<FILE>)
    func load(from file: UnsafeMutablePointer<FILE>)
}
