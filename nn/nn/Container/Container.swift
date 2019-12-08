//
//  Container.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Container {
    func forward(_ input: [Float]) -> [Float]
    func backward(_ label: [Float], rate: Float, derivative: [Float])
    func loss(_ label: [Float]) -> Float
}
