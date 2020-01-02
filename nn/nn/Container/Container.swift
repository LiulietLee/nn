//
//  Container.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Container {
    func forward(_ input: NNArray) -> NNArray
    @discardableResult func backward(_ label: NNArray, rate: Float, delta: NNArray) -> NNArray
    func loss(_ label: NNArray) -> Float
    func save(to file: UnsafeMutablePointer<FILE>)
    func load(from file: UnsafeMutablePointer<FILE>)
}
