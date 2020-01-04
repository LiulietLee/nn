//
//  Container.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public protocol Container: Forwardable {
    func loss(_ label: NNArray) -> Float
}
