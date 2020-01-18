//
//  MNISTModel.swift
//  nn
//
//  Created by Liuliet.Lee on 18/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

let mnistModel = Sequential([
    Conv(3, count: 3, padding: 1),
    Conv(3, count: 3, padding: 1),
    Conv(3, count: 3, padding: 1),
    ReLU(),
    MaxPool(),
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    ReLU(),
    MaxPool(),
    Dense(inFeatures: 6 * 7 * 7, outFeatures: 120),
    Dense(inFeatures: 120, outFeatures: 10)
])
