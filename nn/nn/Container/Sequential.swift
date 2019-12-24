//
//  Network.swift
//  nn
//
//  Created by Liuliet.Lee on 12/10/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Sequential: Container {
    private var layers: [Layer] = []
    private var score = NNArray()
    private var input = NNArray()
    
    public var lossClass = Loss.mod2.self
    
    public func add(_ layer: Layer) {
        layers.append(layer)
    }
    
    public func add(_ layers: [Layer]) {
        self.layers.append(contentsOf: layers)
    }
    
    public func forward(_ input: NNArray) -> NNArray {
        self.input = input.copy()
        var input = input.copy()
        for l in layers {
            autoreleasepool {
                input = l.forward(input)
            }
        }
        score = input.copy()
        return score
    }
    
    public func backward(_ label: NNArray, rate: Float = 0.1, delta: NNArray = NNArray()) {
        var r = delta.isEmpty
            ? lossClass.delta(score: score, label: label)
            : delta
        for i in (0..<layers.count).reversed() {
            autoreleasepool {
                if i == 0 {
                    r = layers[i].backward(input, delta: r, rate: rate)
                } else {
                    r = layers[i].backward(layers[i - 1].score, delta: r, rate: rate)
                }
            }
        }
    }
    
    public func loss(_ label: NNArray) -> Float {
        return lossClass.loss(score: score, label: label)
    }
}
