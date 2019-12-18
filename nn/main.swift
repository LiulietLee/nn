//
//  main.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

let net = Sequential()

/*
net.add([
    Dense(inFeatures: 5, outFeatures: 3),
    Dense(inFeatures: 3, outFeatures: 2)
])

for i in 0..<20 {
    let img = NNArray([1, 0, 0, 0, 0], d: [5])
    let label = NNArray([1, 0], d: [2])
    let score = net.forward(img)
    let loss = net.loss(label)
    net.backward(label, rate: 0.01)

    print("epoch \(i): loss: \(loss), score: \(score.map() { $0 })")
}
*/

net.add([
    Conv(2, 2, count: 3),
    Conv(2, 2),
    Dense(inFeatures: 4, outFeatures: 4)
])

for i in 0..<200 {
    let img = NNArray((0..<16).map{ _ in Float(1) }, d: [4, 4, 1])
    let label = NNArray([1, 0, 0, 0], d: [4])
    let score = net.forward(img)
    let loss = net.loss(label)
    net.backward(label, rate: 0.01)

    print("epoch \(i): loss: \(loss), score: \(score.map() { $0 })")
}
