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
    Dense(inFeatures: 5, outFeatures: 3),
    Dense(inFeatures: 3, outFeatures: 2)
])

for i in 0..<20 {
    let img = NNArray([1, 0, 0, 0, 0], d: [5])
    let label = NNArray([1, 0], d: [2])
    let score = net.forward(img)
    let loss = net.loss(label)
    net.backward(label, rate: 0.1)

    print("epoch \(i): loss: \(loss), score: \(score.map() { $0 })")
}

/*
let a = NNArray(5, 5, 3, initValue: 1)
let conv = Conv(2, 2, count: 3)
conv.setDepth(a.d[2])
let b = conv.forward(a)
print(b.d)
for val in b {
    print(val)
}
*/
