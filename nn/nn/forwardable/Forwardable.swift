//
//  Forwardable.swift
//  nn
//
//  Created by Liuliet.Lee on 4/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Forwardable {
    func forward(_ input: NNArray) -> NNArray
    func predict(_ input: NNArray) -> NNArray
    @discardableResult
    func backward(_ label: NNArray, delta: NNArray) -> NNArray
    func step(lr: Float, momentum: Float)
    func zeroGrad()
    func save(to file: UnsafeMutablePointer<FILE>)
    func load(from file: UnsafeMutablePointer<FILE>)
}
