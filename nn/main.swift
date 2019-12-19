//
//  main.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

let net = Sequential()

net.add([
    Conv(2, 2, count: 3, step: 2, padding: 1),
    Conv(2, 2),
    Dense(inFeatures: 4, outFeatures: 4)
])

let img = NNArray((0..<4 * 4 * 3).map{ _ in Float.random(in: 0...1) }, d: [4, 4, 3])
let label = NNArray([1, 0, 0, 0], d: [4])

for i in 0..<120 {
    let score = net.forward(img)
    let loss = net.loss(label)
    net.backward(label, rate: 0.1)

    print("epoch \(i): loss: \(loss), score: \(score.map() { $0 })")
}
