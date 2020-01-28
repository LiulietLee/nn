//
//  Cifar10Reader.swift
//  nn
//
//  Created by Liuliet.Lee on 29/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import CoreImage

/**
The CIFAR-10 dataset reader.
*/
public class Cifar10Reader: ImageReader {
    
    var rootPath = String()
    var trainingSet = [String]()
    var testSet = [String]()
    var labels = [String]()
    
    var trainIndex = 0
    var testIndex = 0
    
    var batchSize = 0
    
    public enum SetType {
        case train
        case test
    }
    
    /**
     - Description: Expected file structure:
     ```
     |-- root
     |   |-- labels.txt
     |   |-- test
     |   |   |-- 0_cat.png
     |   |   |-- 1_ship.png
     |   |   |-- ...
     |   |-- train
     |   |   |-- 0_frog.png
     |   |   |-- 1_truck.png
     |   |   |-- ...
     ```
     */
    public init(root: String, batchSize: Int) {
        self.batchSize = batchSize
        rootPath = root
        trainingSet = try! ImageReader.fileManager.contentsOfDirectory(atPath: root + "/train")
        testSet = try! ImageReader.fileManager.contentsOfDirectory(atPath: root + "/test")
        labels = try! String(contentsOf: URL(fileURLWithPath: root + "/labels.txt"), encoding: .utf8).split(separator: "\n").map { String($0) }
    }
    
    public func printStatus() {
        print("trainning set size: \(trainingSet.count)")
        print("test set size: \(testSet.count)")
        print("labels: \(labels)")
    }
    
    func getLabel(at index: Int, from type: SetType = .train) -> [Float] {
        let title = String(
            (type == .train ? trainingSet[index] : testSet[index])
                .split(separator: "_")[1]
                .split(separator: ".")[0]
        )
        return labels.map { Float($0 == title ? 1.0 : 0.0) }
    }
    
    public func getTrain(_ index: Int) -> (image: NNArray, label: NNArray)? {
        if index >= trainingSet.count {
            return nil
        }

        let path = rootPath + "/train/" + trainingSet[index]
        let labelArray = getLabel(at: index, from: .train)
        if let image = readImage(path: path) {
            return (image: image, label: NNArray(labelArray, d: [1, labelArray.count]))
        } else {
            return nil
        }
    }

    public func getTest(_ index: Int) -> (image: NNArray, label: Int)? {
        if index >= testSet.count {
            return nil
        }
        
        let path = rootPath + "/test/" + testSet[index]
        let label = getLabel(at: index, from: .test).firstIndex(of: 1.0)!
        if let image = readImage(path: path) {
            return (image: image, label: label)
        } else {
            return nil
        }
    }
    
    public func nextTrain() -> (image: NNArray, label: NNArray)? {
        if trainIndex + batchSize - 1 >= trainingSet.count {
            trainIndex = 0
            return nil
        }
        
        let imageholder = NNArray(batchSize, 3, 32, 32)
        let labelholder = NNArray(batchSize, labels.count)

        let pool = ThreadPool(count: batchSize)
        pool.run { (i) in
            let path = self.rootPath + "/train/" + self.trainingSet[self.trainIndex + i]
            _ = self.readImage(path: path, buffer: imageholder, batch: i)
            let label = self.getLabel(at: self.trainIndex + i, from: .train)
            for j in 0..<self.labels.count {
                labelholder[i, j] = label[j]
            }
        }
        trainIndex += batchSize
        
        return (imageholder, labelholder)
    }
    
    public func nextTest() -> (image: NNArray, label: Int)? {
        if testIndex >= testSet.count {
            testIndex = 0
            return nil
        }
        
        let imageholder = NNArray(1, 3, 32, 32)

        let path = rootPath + "/test/" + testSet[testIndex]
        _ = readImage(path: path, buffer: imageholder, batch: 1)
        let label = getLabel(at: testIndex, from: .test).firstIndex(of: 1.0)!
        
        testIndex += 1
        
        return (imageholder, label)
    }
}
