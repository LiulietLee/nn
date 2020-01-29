//
//  DatasetReader.swift
//  nn
//
//  Created by Liuliet.Lee on 28/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public class DatasetReader {
    
    typealias File = UnsafeMutablePointer<FILE>
    var batchSize = 0

    public enum SetType {
        case train
        case test
    }
}
