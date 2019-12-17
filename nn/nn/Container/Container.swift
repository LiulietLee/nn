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
    func backward(_ label: NNArray, rate: Float, derivative: NNArray)
    func loss(_ label: NNArray) -> Float
}
