//
//  MNISTReader.swift
//  nn
//
//  Created by Liuliet.Lee on 18/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public class MNISTReader: ImageReader {
    
    var rootPath = String()
    
    var trainImage = NNArray()
    var trainLabel = NNArray()
    var trainSetSize: Int { trainImage.d[0] }
    
    var testImage = NNArray()
    var testLabel = [Int]()
    var testSetSize: Int { testImage.d[0] }
    var trainIndex = 0
    var testIndex = 0
    
    var batchSize = 0
    
    typealias File = UnsafeMutablePointer<FILE>
    
    var trainDataFile: File
    var trainLabelFile: File
    var testDataFile: File
    var testLabelFile: File
    
    var trainDataLoadIndex = 0
    var testDataLoadIndex = 0

    public enum SetType {
        case train
        case test
    }
    
    /**
     - Description: Expected file structure:
     ```
     |-- root
     |   |-- t10k-images-idx3-ubyte
     |   |-- t10k-labels-idx1-ubyte
     |   |-- train-images-idx3-ubyte
     |   |-- train-labels-idx1-ubyte
     ```
     */
    public init(root: String, batchSize: Int) {
        self.batchSize = batchSize
        rootPath = root
        
        testLabelFile    = fopen(rootPath + "/t10k-labels-idx1-ubyte", "r")
        testDataFile     = fopen(rootPath + "/t10k-images-idx3-ubyte", "r")
        trainLabelFile   = fopen(rootPath + "/train-labels-idx1-ubyte", "r")
        trainDataFile    = fopen(rootPath + "/train-images-idx3-ubyte", "r")
        
        super.init()
        dataInit()
    }
    
    private func dataInit() {
        if !checkTrainingData() || !checkTestData() {
            fatalError("data file broken")
        }
    }

    public func getTrain(_ index: Int) -> (image: NNArray, label: NNArray)? {
        if index >= trainSetSize {
            return nil
        }

        while trainDataLoadIndex < index {
            readData(&trainImage, &trainDataLoadIndex, file: trainDataFile)
        }
        
        let img = trainImage.subArray(at: index)
        let label = trainLabel.subArray(at: index)
        return (img, label)
    }

    public func getTest(_ index: Int) -> (image: NNArray, label: Int)? {
        if index >= testSetSize {
            return nil
        }
        
        while testDataLoadIndex < index {
            readData(&testImage, &testDataLoadIndex, file: testDataFile)
        }

        let img = testImage.subArray(at: index)
        let label = testLabel[index]
        return (img, label)
    }

    public func nextTrain() -> (image: NNArray, label: NNArray)? {
        if trainIndex + batchSize - 1 >= trainSetSize {
            trainIndex = 0
            return nil
        }
        
        while trainDataLoadIndex < trainIndex + batchSize {
            readData(&trainImage, &trainDataLoadIndex, file: trainDataFile)
        }

        let img = trainImage.subArray(
            at: trainIndex,
            length: batchSize * trainImage.acci[0]
        )
        let label = trainLabel.subArray(
            at: trainIndex,
            length: batchSize * trainLabel.acci[0]
        )
        trainIndex += batchSize
        
        return (img, label)
    }

    public func nextTest() -> (image: NNArray, label: Int)? {
        if testIndex >= testSetSize {
            testIndex = 0
            return nil
        }

        defer {
            testIndex += 1
        }
        
        return getTest(testIndex)
    }
    
    private func checkTrainingData() -> Bool {
        let labelFile = trainLabelFile
        let dataFile = trainDataFile

        if magicNumberCheck(dataFile: dataFile, labelFile: labelFile) {
            readImageMeta(&trainImage, file: dataFile)
            var labels = [Int]()
            readLabel(&labels, file: labelFile)
            trainLabel = NNArray(labels.count, 10)
            for i in 0..<labels.count {
                for j in 0..<10 {
                    trainLabel[i, j] = labels[i] == j ? 1.0 : 0.0
                }
            }
            return true
        } else {
            return false
        }
    }
    
    private func checkTestData() -> Bool {
        let labelFile = testLabelFile
        let dataFile = testDataFile

        if magicNumberCheck(dataFile: dataFile, labelFile: labelFile) {
            readImageMeta(&testImage, file: dataFile)
            readLabel(&testLabel, file: labelFile)
            
            return true
        } else {
            return false
        }
    }
    
    private func magicNumberCheck(dataFile: File, labelFile: File) -> Bool {
        var magic: UInt32 = 0
        fread(&magic, 4, 1, labelFile)
        reverseInt(&magic)
        if magic != 2049 { return false }
        
        fread(&magic, 4, 1, dataFile)
        reverseInt(&magic)
        if magic != 2051 { return false }
        
        return true
    }
    
    private func readImageMeta(_ buf: inout NNArray, file f: File) {
        var size: UInt32 = 0
        fread(&size, 4, 1, f)
        reverseInt(&size)

        var row: UInt32 = 0, col: UInt32 = 0

        fread(&row, 4, 1, f)
        fread(&col, 4, 1, f)
        reverseInt(&row)
        reverseInt(&col)

        buf = NNArray(Int(size), 1, Int(row), Int(col))
    }
    
    private func readData(_ buf: inout NNArray, _ idx: inout Int, file f: File) {
        let row = buf.d[2]
        let col = buf.d[3]
        
        var pix = [UInt8](repeating: 0, count: row * col)

        fread(&pix, 1, row * col, f)
        for i in 0..<row {
            for j in 0..<col {
                buf[idx, 0, i, j] = Float(pix[i * col + j]) / 255.0
            }
        }
        idx += 1
    }
    
    private func readLabel(_ buf: inout [Int], file f: File) {
        var size: UInt32 = 0
        fread(&size, 4, 1, f)
        reverseInt(&size)
        var raw = [UInt8](repeating: 0, count: Int(size))
        fread(&raw, 1, Int(size), f)
        buf = raw.map { Int($0) }
    }
    
    private func reverseInt(_ i: inout UInt32) {
        var c1: UInt32, c2: UInt32, c3: UInt32, c4: UInt32

        c1 = i & 255
        c2 = (i >> 8) & 255
        c3 = (i >> 16) & 255
        c4 = (i >> 24) & 255

        i = (c1 << 24) + (c2 << 16) + (c3 << 8) + c4
    }
}
