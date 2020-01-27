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

let reader = MNISTReader(
    root: "/Users/liulietlee/Developer/tf/mnist",
    batchSize: 16
)

let net = mnistModel

func train() {
    ModelStorage.load(net, path: "mnistmodel04.nnm")

    for i in 0..<10 {
        autoreleasepool {
            var runningLoss: Float = 0.0
//            reader.trainIndex = 0
            while let (img, label) = reader.nextTrain() {
                net.zeroGrad()
                let _ = net.forward(img)
                let loss = net.loss(label)
                net.backward(label)
                net.step(lr: 0.00008, momentum: 0.99)
                runningLoss = max(runningLoss, loss)
                
                if reader.trainIndex % 1600 == 1584 {
                    print("[\(i), \(reader.trainIndex)] loss: \(runningLoss)")
//                    print(score)
                    runningLoss = 0.0
                    ModelStorage.save(net, path: "mnistmodel04.nnm")
//                    break
                }
            }
        }
    }
    
    ModelStorage.save(net, path: "mnistmodel04.nnm")
}

func test() {
    ModelStorage.load(net, path: "mnistmodel.nnm")
    
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

//train()
test()
