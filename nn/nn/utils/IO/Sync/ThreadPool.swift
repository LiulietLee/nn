//
//  ThreadPool.swift
//  nn
//
//  Created by Liuliet.Lee on 6/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

class ThreadPool {
    var mutexQueue = [DispatchSemaphore]()
    var count = 0
    
    init(count: Int) {
        self.count = count
        for _ in 0..<count {
            mutexQueue.append(DispatchSemaphore(value: 1))
        }
    }
    
    func run(_ task: @escaping ((Int) -> Void)) {
        for i in 0..<count {
            mutexQueue[i].wait()
        }
        for i in 0..<count {
            DispatchQueue.global().async {
                task(i)
                self.mutexQueue[i].signal()
            }
        }
        for i in 0..<count {
            mutexQueue[i].wait()
            mutexQueue[i].signal()
        }
    }
}
