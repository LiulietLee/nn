//
//  AutoRunner.swift
//  nn
//
//  Created by Liuliet.Lee on 16/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public class AutoRunner {
    
    var customer: ((_ input: NNArray, _ label: NNArray?) -> Void)
    var provider: (() -> (input: NNArray, label: NNArray?)?)
    var epoch: Int
    
    let mutex = DispatchSemaphore(value: 1)
    let full = DispatchSemaphore(value: 0)
    let empty = DispatchSemaphore(value: 1)
    
    var inputPlaceholder = NNArray()
    var labelPlaceholder = NNArray()
    var runningEpoch = 0

    public init(
        epoch: Int,
        dataCustomer: @escaping ((_ input: NNArray, _ label: NNArray?) -> Void),
        dataProvider: @escaping (() -> (input: NNArray, label: NNArray?)?)) {
        self.epoch = epoch
        customer = dataCustomer
        provider = dataProvider
    }
    
    public func run() {
        let pool = ThreadPool(count: 2)
        runningEpoch = 0
        
        pool.run { i in
            if i == 0 { // customer
                while self.runningEpoch < self.epoch {
                    self.customRun()
                }
            } else {    // provider
                while self.runningEpoch < self.epoch {
                    self.providerRun()
                }
            }
        }
    }
    
    private func customRun() {
        full.wait()
        
        mutex.wait()
        let input: NNArray = inputPlaceholder.copy()
        let label: NNArray = labelPlaceholder.copy()
        mutex.signal()
        
        customer(input, label)
        
        empty.signal()
    }
    
    private func providerRun() {
        empty.wait()
        
        if let (input, label) = provider() {
            mutex.wait()
            inputPlaceholder = input
            if label != nil {
                labelPlaceholder = label!
            }
            mutex.signal()
        } else {
            runningEpoch += 1
        }

        full.signal()
    }
}
