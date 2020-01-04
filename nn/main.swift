//
//  main.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

Core.device = MTLCreateSystemDefaultDevice()

var net = Sequential()

net.add([
    Conv(2, count: 3),
    Conv(2, count: 3),
    Conv(3),
    MaxPooling(),
    Dense(inFeatures: 9, outFeatures: 5),
    Dense(inFeatures: 5, outFeatures: 5),
    Dense(inFeatures: 5, outFeatures: 2)
])

let img = NNArray(10, 10, 3, initValue: 1.0)
let label = NNArray([1.0, 0.0])

func train() {
    for i in 0..<20 {
        let score = net.forward(img)
        let loss = net.loss(label)
        net.backward(label, rate: 0.1)
        
        print("\(i) loss: \(loss) score: \(score.map { $0 })")
    }
    
    ModelStorage.save(net, path: "a.txt")
}

func test() {
    ModelStorage.load(net, path: "a.txt")
    
    let score = net.forward(img)
    let loss = net.loss(label)
    print("test: - loss: \(loss) score: \(score.map { $0 })")
}

test()
//train()

/*
 let reader = Cifar10Reader(root: "/Users/liulietlee/Developer/tf/cifar/cifar")

print("train.")

for i in 0..<120 {
    var runningLoss = 0.0
    
    for index in 0..<20 {
        autoreleasepool {
            let (input, n) = reader.getTest(index)!
            let label = NNArray((0..<10).map { $0 == n ? 1.0 : 0.0 })
            let _ = net.forward(input)
            let loss = net.loss(label)
            net.backward(label, rate: 0.001)
            runningLoss += Double(loss)
        }
    }

    print("[\(i)] loss: \(runningLoss / 20)")
    runningLoss = 0.0
}

print("test.")

var count = 0
for index in 0..<20 {
    let (input, label) = reader.getTest(index)!
    let score = net.forward(input)
    var maxi = 0
    for i in 1..<score.count {
        if score[i] > score[maxi] {
            maxi = i
        }
    }
    print("[\(index)] pred: \(maxi), label: \(label)")
    print(score.map{ $0 })
    if maxi == label {
        count += 1
    }
}

print("\(count)/20")
*/
