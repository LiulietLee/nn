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
    private var score = [Float]()
    private var input = [Float]()
    
    public var lossClass = Loss.mod2.self
    
    public func add(_ layer: Layer) {
        layers.append(layer)
    }
    
    public func add(_ layers: [Layer]) {
        self.layers.append(contentsOf: layers)
    }
    
    public func forward(_ input: [Float]) -> [Float] {
        self.input = input
        var input = input
        for l in layers {
            input = l.forward(input)
        }
        score = input
        return score
    }
    
    public func backward(_ label: [Float], rate: Float = 0.1, derivative: [Float] = []) {
        var r = derivative.isEmpty
            ? lossClass.derivative(score: score, label: label)
            : derivative
        for i in (0..<layers.count).reversed() {
            if i == 0 {
                r = layers[i].backward(input, derivative: r, rate: rate)
            } else {
                r = layers[i].backward(layers[i - 1].score, derivative: r, rate: rate)
            }
        }
    }
    
    public func loss(_ label: [Float]) -> Float {
        return lossClass.loss(score: score, label: label)
    }
}
