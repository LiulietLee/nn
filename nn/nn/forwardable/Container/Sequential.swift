//
//  Network.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

/**
 A Sequential Container.
 
 Example:
 ```
 let model = Sequential([
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
 
 let score = model.forward(input)
 ```
 */
public class Sequential: BaseContainer {
    @objc dynamic var layers: [BaseLayer] = []
    var score = NNArray()
    var input = NNArray()
    
    /**
     Specify how to calculate loss and derivative.
     */
    public var lossClass: AbstractLoss.Type = Loss.svm.self
    
    public override init() {
        super.init()
    }
    
    public init(_ layers: [BaseLayer]) {
        super.init()
        add(layers)
    }
    
    public func add(_ layer: BaseLayer) {
        layers.append(layer)
    }
    
    public func add(_ layers: [BaseLayer]) {
        self.layers.append(contentsOf: layers)
    }
    
    public override func forward(_ input: NNArray) -> NNArray {
        self.input = input.copy()
        var input: NNArray = input.copy()
        for l in layers {
            autoreleasepool {
                input = l.forward(input)
            }
        }
        score = input.copy()
        return score
    }
    
    public override func predict(_ input: NNArray) -> NNArray {
        var input = input
        for l in layers {
            autoreleasepool {
                input = l.predict(input)
            }
        }
        score = input
        return score
    }
    
    @discardableResult
    public override func backward(_ label: NNArray, delta: NNArray = NNArray()) -> NNArray {
        var r = delta.isEmpty
            ? lossClass.delta(score: score, label: label)
            : delta
        for i in (0..<layers.count).reversed() {
            autoreleasepool {
                if i == 0 {
                    r = layers[i].backward(input, delta: r)
                } else {
                    r = layers[i].backward(layers[i - 1].score, delta: r)
                }
            }
        }
        return r
    }
    
    public override func zeroGrad() {
        let pool = ThreadPool(count: layers.count)
        pool.run { i in
            self.layers[i].zeroGrad()
        }
    }
    
    public override func step(lr: Float, momentum: Float) {
        let pool = ThreadPool(count: layers.count)
        pool.run { i in
            self.layers[i].step(lr: lr, momentum: momentum)
        }
    }
    
    public override func loss(_ label: NNArray) -> Float {
        return lossClass.loss(score: score, label: label)
    }
}
