//
//  main.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import Metal

Core.device = MTLCreateSystemDefaultDevice()

let net = Sequential()

net.add([
    Conv(2, 2, count: 3, step: 2),
    Conv(2, 2, step: 2),
    Conv(2, 2, count: 3, step: 2),
    Conv(2, 2, step: 2)
])

//net.add([
//    Conv(2, 2, step: 2)
//])

let img = NNArray(32, 32, 3, initValue: 1.0)
let label = NNArray([1, 1, 1, 1], d: [4])

let start = DispatchTime.now()

for i in 0..<10 {
    let score = net.forward(img)
    let loss = net.loss(label)
    net.backward(label, rate: 0.1)

    print("epoch \(i): loss: \(loss), score: \(score.map() { $0 })")
}

let end = DispatchTime.now()

let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
let timeInterval = Double(nanoTime) / 1_000_000_000

print("\(timeInterval) seconds")

/*
let reader = ImageReader()

let basePath = "/Users/liulietlee/Desktop/test/"

func gen(_ path: String, _ label: Int) -> (NNArray, NNArray) {
    let realPath = basePath + path
    guard let img = reader.cifar10Image(path: realPath) else {
        exit(EXIT_FAILURE)
    }
    let labels = NNArray((0..<10).map() { $0 == label ? 1.0: 0.0 }, d: [10])
    return (img, labels)
}

let trainSet = [
    gen("0_cat.png", 3),
    gen("1_ship.png", 9),
    gen("2_ship.png", 9),
    gen("3_airplane.png", 0),
    gen("4_frog.png", 6),
    gen("5_frog.png", 6),
    gen("6_automobile.png", 1),
    gen("7_frog.png", 6),
    gen("8_cat.png", 3),
    gen("9_automobile.png", 1)
]

let testSet = [
    gen("10_airplane.png", 0),
    gen("15_ship.png", 9),
    gen("19_frog.png", 6)
]

for i in 0..<50 {
    var loss = Float()
    
    for (img, label) in trainSet {
        let start = DispatchTime.now()
        let _ = net.forward(img)
        loss += net.loss(label)
        net.backward(label, rate: 0.1)
        let end = DispatchTime.now()
        
        let nanoTime = end.uptimeNanoseconds - start.uptimeNanoseconds
        let timeInterval = Double(nanoTime) / 1_000_000_000

        print("\(timeInterval) seconds")
    }

    print("epoch \(i): loss: \(loss / Float(trainSet.count))")
}

for (img, _) in testSet {
    let score = net.forward(img)
    print(score)
    var idx = 0
    for i in 0..<score.count {
        if score[i] > score[idx] {
            idx = i
        }
    }
    print(idx)
}
*/
