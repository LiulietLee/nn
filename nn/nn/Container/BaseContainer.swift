//
//  BaseContainer.swift
//  nn
//
//  Created by Liuliet.Lee on 30/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class BaseContainer: NSObject, Container {
    public func forward(_ input: NNArray) -> NNArray {
        return input
    }
    
    @discardableResult
    public func backward(_ label: NNArray, rate: Float, delta: NNArray) -> NNArray {
        return delta
    }
    
    public func loss(_ label: NNArray) -> Float {
        return 0.0
    }
        
    public func save(to file: UnsafeMutablePointer<FILE>) {
        ModelStorage.save(self, file: file)
    }
    
    public func load(from file: UnsafeMutablePointer<FILE>) {
        ModelStorage.load(self, file: file)
    }
}
