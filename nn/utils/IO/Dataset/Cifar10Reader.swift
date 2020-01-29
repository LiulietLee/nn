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
public class Cifar10Reader: DatasetReader {
    
    var rootPath = String()
    
    var trainImage = NNArray()
    var trainLabel = NNArray()
    var trainSetSize: Int { trainImage.d[0] }
    
    var testImage = NNArray()
    var testLabel = NNArray()
    var testSetSize: Int { testImage.d[0] }
    var trainIndex = 0
    var testIndex = 0
            
    var trainDataFile: File
    var testDataFile: File
    
    var trainDataFileIndex = 1
    var trainDataLoadIndex = 0
    var testDataLoadIndex = 0
    
    /**
     Set the path to Cifar-10 dataset and batch size.
     
     Recommended data download method:
     ```
     curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
     ```
     
     Expected file structure:
     ```
     |-- root
     |   |-- data_batch_1.bin
     |   |-- data_batch_2.bin
     |   |-- data_batch_3.bin
     |   |-- data_batch_4.bin
     |   |-- data_batch_5.bin
     |   |-- test_batch.bin
     |   |-- batches.meta.txt
     ```
     */
    public init(root: String, batchSize: Int) {
        rootPath = root
        
        testDataFile     = fopen(rootPath + "/data_batch_1.bin", "r")
        trainDataFile    = fopen(rootPath + "/test_batch.bin", "r")
        
        trainImage = NNArray(50000, 3, 32, 32)
        testImage = NNArray(10000, 3, 32, 32)
        trainLabel = NNArray(50000, 10)
        testLabel = NNArray(10000, 10)
        
        super.init()
        self.batchSize = batchSize
    }

    public func getTrain(_ index: Int) -> (image: NNArray, label: NNArray)? {
        if index >= trainSetSize {
            return nil
        }

        while trainDataLoadIndex <= index {
            readNextTrainData()
        }
        
        let img = trainImage.subArray(at: index)
        let label = trainLabel.subArray(at: index)
        return (img, label)
    }

    public func getTest(_ index: Int) -> (image: NNArray, label: NNArray)? {
        if index >= testSetSize {
            return nil
        }
        
        while testDataLoadIndex <= index {
            readNextTestData()
        }

        let img = testImage.subArray(at: index)
        let label = testLabel.subArray(at: index)
        return (img, label)
    }

    public func nextTrain() -> (image: NNArray, label: NNArray)? {
        if trainIndex + batchSize - 1 >= trainSetSize {
            trainIndex = 0
            return nil
        }
        
        while trainDataLoadIndex < trainIndex + batchSize {
            readNextTrainData()
        }

        let img = trainImage.subArray(
            at: trainIndex,
            length: batchSize * trainImage.acci[0],
            d: [batchSize, 3, trainImage.d[2], trainImage.d[3]]
        )
        let label = trainLabel.subArray(
            at: trainIndex,
            length: batchSize * trainLabel.acci[0],
            d: [batchSize, trainLabel.d[1]]
        )
        trainIndex += batchSize
        
        return (img, label)
    }

    public func nextTest() -> (image: NNArray, label: NNArray)? {
        if testIndex >= testSetSize {
            testIndex = 0
            return nil
        }

        defer {
            testIndex += 1
        }
        
        return getTest(testIndex)
    }
    
    private func readNextTrainData() {
        readData(trainImage, trainLabel, &trainDataLoadIndex, file: trainDataFile)
        if trainDataLoadIndex % 10000 == 9999, trainDataFileIndex < 5 {
            trainDataFileIndex += 1
            trainDataFile = fopen(rootPath + "/data_batch_\(trainDataFileIndex).bin", "r")
        }
    }
    
    private func readNextTestData() {
        readData(testImage, testLabel, &testDataLoadIndex, file: testDataFile)
    }
    
    private func readData(_ buf: NNArray, _ la: NNArray, _ idx: inout Int, file f: File) {
        var pix = [UInt8](repeating: 0, count: 3072)
        var label: UInt8 = 0

        fread(&label, 1, 1, f)
        for i in 0..<10 {
            la[idx, i] = i == Int(label) ? 1.0 : 0.0
        }
        
        fread(&pix, 1, 3072, f)
        var i = 0
        for c in 0..<3 {
            for row in 0..<32 {
                for col in 0..<32 {
                    buf[idx, c, row, col] = Float(pix[i]) / 255.0
                    i += 1
                }
            }
        }
        
        idx += 1
    }
}
