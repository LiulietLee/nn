//
//  MNISTModel.swift
//  nn
//
//  Created by Liuliet.Lee on 18/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

let mnistmodel1 = Sequential([
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    ReLU(),
    MaxPool(2, step: 2),
    Conv(3, count: 9, padding: 1),
    Conv(3, count: 9, padding: 1),
    Conv(3, count: 9, padding: 1),
    ReLU(),
    MaxPool(2, step: 2),
    Dense(inFeatures: 9 * 7 * 7, outFeatures: 120),
    Dense(inFeatures: 120, outFeatures: 10)
])

let mnistModel = mnistmodel1
