//
//  MNISTModel.swift
//  nn
//
//  Created by Liuliet.Lee on 18/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

let mnistmodel1 = Sequential([
    Conv(3, count: 3, padding: 1),
    Conv(3, count: 3, padding: 1),
    Conv(3, count: 3, padding: 1),
    ReLU(),
    MaxPool(2, step: 2),
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    Conv(3, count: 6, padding: 1),
    ReLU(),
    MaxPool(2, step: 2),
    Dense(inFeatures: 6 * 7 * 7, outFeatures: 120),
    Dense(inFeatures: 120, outFeatures: 10)
])

let mnistmodel2 = Sequential([
    Conv(3, count: 3, padding: 1),
    ReLU(),
    Inception([
        [Conv(1, count: 6)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 6, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 6)]
    ]),
    ReLU(),
    Inception([
        [Conv(1, count: 6)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 6, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 6)]
    ]),
    ReLU(),
    MaxPool(7, step: 7),
    Dense(inFeatures: 24 * 4 * 4, outFeatures: 10),
])

let mnistmodel3 = Sequential([
    Conv(3, count: 3, padding: 1),
    ReLU(),
    Inception([
        [Conv(1, count: 6)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 6, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 6)]
    ]),
    ReLU(),
    Inception([
        [Conv(1, count: 6)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 6, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 6)]
    ]),
    ReLU(),
    Conv(7, count: 24, step: 7),
    Dense(inFeatures: 24 * 4 * 4, outFeatures: 10),
])

let mnistmodel4 = Sequential([
    Conv(3, count: 3, padding: 1),
    ReLU(),
    Inception([
        [Conv(1, count: 3)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 9, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 3)]
    ]),
    ReLU(),
    Inception([
        [Conv(1, count: 3)],
        [Conv(3, count: 6, padding: 1)],
        [Conv(5, count: 9, padding: 2)],
        [MaxPool(3, padding: 1), Conv(1, count: 3)]
    ]),
    ReLU(),
    AveragePool(7),
    Dense(inFeatures: 21 * 4 * 4, outFeatures: 10),
])

let mnistModel = mnistmodel4
