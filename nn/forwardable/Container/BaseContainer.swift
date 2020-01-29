//
//  BaseContainer.swift
//  nn
//
//  Created by Liuliet.Lee on 30/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class BaseContainer: NSObject, Container {
    
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
     
     - parameter label: Correct label.
     - parameter delta: Derivative from the back container of the current container.
     
     - returns: Returned derivative should be passed to the previous container. If there aren't previous container, just ignore the return value.
     */
    public func backward(_ label: NNArray, delta: NNArray = NNArray()) -> NNArray {
        return delta
    }
    
    /**
     Update parameters of current container.
     
     `nn` use the simplified Adam algorithm to update parameters. The full Adam update is a little bit complicated and I am too lazy to implement it.
     
     Gradient **won't** be set to zero after execting this method.
     
     - parameter lr: Learning rate.
     - parameter momentum: Influences the velocity
     */
    public func step(lr: Float, momentum: Float) {}
    
    /**
     Set gradient to be zero.
     */
    public func zeroGrad() {}
    
    /**
     Loss function.
     
     - parameter label: Correct label.
     - returns: Loss value.
     */
    public func loss(_ label: NNArray) -> Float {
        return 0.0
    }
    
    /**
     Write dynamic contents of current container to the disk.
     
     - parameter file: Unsafe pointer to the file.
     */
    public func save(to file: UnsafeMutablePointer<FILE>) {
        ModelStorage.save(self, file: file)
    }
    
    /**
     Read dynamic contents of current container from the disk.
     
     - parameter file: Unsafe pointer to the file.
     */
    public func load(from file: UnsafeMutablePointer<FILE>) {
        ModelStorage.load(self, file: file)
    }
}
