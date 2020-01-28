//
//  BaseLayer.swift
//  nn
//
//  Created by Liuliet.Lee on 30/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class BaseLayer: NSObject, Layer {

    /**
     Store the output of this layer.
     */
    public var score = NNArray()
    var batchSize = 0
    
    /**
     Call this function to do prediction.
     */
    public func forward(_ input: NNArray) -> NNArray {
        return input
    }
    
    /**
     Not recommended to use this.
     */
    public func predict(_ input: NNArray) -> NNArray {
        return forward(input)
    }
    
    /**
     Compute derivative.
     
     This function won't change the parameters of current layer. If you wanna update parameters, you need to call `step()` method.
     
     - parameter input: Should be the same value as the forward one.
     - parameter delta: Derivative from the back layer of the current layer.
     
     - returns: Returned derivative should be passed to the previous layer.
     */
    public func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        return delta
    }
    
    /**
     Set gradient to be zero.
     */
    public func zeroGrad() {}
    
    /**
     Update parameters of current layer.
     
     `nn` use the simplified Adam algorithm to update parameters. The full Adam update is a little bit complicated and I am too lazy to implement it.
     
     Gradient **won't** be set to zero after execting this method.
     
     - parameter lr: Learning rate.
     - parameter momentum: Influences the velocity
     */
    public func step(lr: Float, momentum: Float) {}
    
    /**
     Write dynamic contents of current layer to the disk.
     
     - parameter file: Unsafe pointer to the file.
     */
    public func save(to file: UnsafeMutablePointer<FILE>) {
        ModelStorage.save(self, file: file)
    }
    
    /**
     Read dynamic contents of current layer from the disk.
     
     - parameter file: Unsafe pointer to the file.
     */
    public func load(from file: UnsafeMutablePointer<FILE>) {
        ModelStorage.load(self, file: file)
    }
}
