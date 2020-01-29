//
//  main.swift
//  mnist-mac
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

Core.device = MTLCreateSystemDefaultDevice()

let reader = MNISTReader(
    root: "/Users/liulietlee/Developer/tf/mnist",
    batchSize: 16
)

let net = Sequential([
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

func train() {
    ModelStorage.load(net, path: "mnistmodel01.nnm")

    for i in 0..<10 {
        var j = 0
        var runningLoss: Float = 0.0
        while let (img, label) = reader.nextTrain() {
            net.zeroGrad()
            let _ = net.forward(img)
            let loss = net.loss(label)
            net.backward(label)
            net.step(lr: 0.0001, momentum: 0.99)
            runningLoss = max(runningLoss, loss)
            
            if j % 10 == 9 {
                print("[\(i), \(j)] loss: \(runningLoss)")
                runningLoss = 0.0
                ModelStorage.save(net, path: "mnistmodel01.nnm")
            }
            
            j += 1
        }
    }
    
    ModelStorage.save(net, path: "mnistmodel01.nnm")
}

func test() {
    ModelStorage.load(net, path: "mnistmodel01.nnm")
    
    var cor = 0
    var tot = 0
    while let (img, label) = reader.nextTest() {
        let score = net.forward(img)
        let pred = score.indexOfMax()
        if pred == label {
            cor += 1
            print("\(tot): Y \(pred) == \(label)")
        } else {
            print("\(tot): N \(pred) != \(label)")
        }
        tot += 1
    }
    print("correct: \(cor) / \(tot)")
}

train()
//test()
