//
//  BaseLayer.swift
//  nn
//
//  Created by Liuliet.Lee on 30/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class BaseLayer: NSObject, Layer {

    public var score = NNArray()
    var batchSize = 0
    
    public func forward(_ input: NNArray) -> NNArray {
        return input
    }
    
    public func predict(_ input: NNArray) -> NNArray {
        return forward(input)
    }
    
    public func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        return delta
    }
    
    public func zeroGrad() {}
    public func step(lr: Float, momentum: Float) {}
        
    public func save(to file: UnsafeMutablePointer<FILE>) {
        ModelStorage.save(self, file: file)
    }
    
    public func load(from file: UnsafeMutablePointer<FILE>) {
        ModelStorage.load(self, file: file)
    }
}
