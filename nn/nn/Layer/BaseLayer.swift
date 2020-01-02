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
    
    public func forward(_ input: NNArray) -> NNArray {
        return input
    }
    
    public func backward(_ input: NNArray, delta: NNArray, rate: Float) -> NNArray {
        return delta
    }
    
    public func save(to file: UnsafeMutablePointer<FILE>) {
        ModelStorage.save(self, file: file)
    }
    
    public func load(from file: UnsafeMutablePointer<FILE>) {
        ModelStorage.load(self, file: file)
    }
}
