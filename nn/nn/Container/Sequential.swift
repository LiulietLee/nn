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
    
    public func backward(_ label: [Float], lr: Float = 0.1) {
        var r = zip(label, score).map { return -2.0 * ($0.0 - $0.1) }
        for i in (0..<layers.count).reversed() {
            if i == 0 {
                r = layers[i].backward(input, derivative: r, lr: lr)
            } else {
                r = layers[i].backward(layers[i - 1].score, derivative: r, lr: lr)
            }
        }
    }
}
