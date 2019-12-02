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
    let img: [Float] = [1, 0, 0, 0, 0]
    let label: [Float] = [1, 0]
    let score = net.forward(img)
    let loss = Loss.mod2(score: score, label: img)
    net.backward(label, rate: 0.1)

    print("epoch \(i): loss: \(loss), score: \(score)")
}
