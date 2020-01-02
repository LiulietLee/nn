//
//  Cifar10Reader.swift
//  nn
//
//  Created by Liuliet.Lee on 29/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import CoreImage

public class Cifar10Reader: ImageReader {
    
    var rootPath = String()
    var trainingSet = [String]()
    var testSet = [String]()
    var labels = [String]()
    
    var trainIndex = 0
    var testIndex = 0
    
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
    public init(root: String) {
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
    
    public func getTrain(_ index: Int) -> (image: NNArray, label: NNArray)? {
        if index >= trainingSet.count {
            return nil
        }
        
        let path = rootPath + "/train/" + trainingSet[index]
        let title = String(
            trainingSet[index].split(separator: "_")[1].split(separator: ".")[0]
        )
        let labelArray = labels.map { Float($0 == title ? 1.0 : 0.0) }
        if let image = readImage(path: path) {
            return (image: image, label: NNArray(labelArray))
        } else {
            return nil
        }
    }

    public func getTest(_ index: Int) -> (image: NNArray, label: Int)? {
        if index >= testSet.count {
            return nil
        }
        
        let path = rootPath + "/test/" + testSet[index]
        let title = String(
            testSet[index].split(separator: "_")[1].split(separator: ".")[0]
        )
        let label = labels.firstIndex { $0 == title }!
        if let image = readImage(path: path) {
            return (image: image, label: label)
        } else {
            return nil
        }
    }
    
    public func nextTrain() -> (image: NNArray, label: NNArray)? {
        if trainIndex >= trainingSet.count {
            trainIndex = 0
            return nil
        }
        
        let path = rootPath + "/train/" + trainingSet[trainIndex]
        let title = String(
            trainingSet[trainIndex].split(separator: "_")[1].split(separator: ".")[0]
        )
        let labelArray = labels.map { Float($0 == title ? 1.0 : 0.0) }
        trainIndex += 1
        if let image = readImage(path: path) {
            return (image: image, label: NNArray(labelArray))
        } else {
            return nil
        }
    }
    
    public func nextTest() -> (image: NNArray, label: Int)? {
        if testIndex >= testSet.count {
            testIndex = 0
            return nil
        }
        
        let path = rootPath + "/test/" + testSet[testIndex]
        let title = String(
            testSet[testIndex].split(separator: "_")[1].split(separator: ".")[0]
        )
        let label = labels.firstIndex { $0 == title }!
        testIndex += 1
        if let image = readImage(path: path) {
            return (image: image, label: label)
        } else {
            return nil
        }
    }
}
