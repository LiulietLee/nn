//
//  Storagable.swift
//  nn
//
//  Created by Liuliet.Lee on 1/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

protocol Storagable {
    func save(to file: UnsafeMutablePointer<FILE>)
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable
}

extension Int: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        var temp = self
        fwrite(&temp, MemoryLayout<Int>.stride, 1, file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        var value = 0
        fread(&value, MemoryLayout<Int>.stride, 1, file)
        return value
    }
}

extension Bool: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        var temp = self
        fwrite(&temp, MemoryLayout<Bool>.stride, 1, file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        var value = true
        fread(&value, MemoryLayout<Bool>.stride, 1, file)
        return value
    }
}

extension Float: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        var temp = self
        fwrite(&temp, MemoryLayout<Float>.stride, 1, file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        var value = Float(0.0)
        fread(&value, MemoryLayout<Float>.stride, 1, file)
        return value
    }
}

extension Double: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        var temp = self
        fwrite(&temp, MemoryLayout<Double>.stride, 1, file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        var value = 0.0
        fread(&value, MemoryLayout<Double>.stride, 1, file)
        return value
    }
}

extension LLVector: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        length.save(to: file)
        capacity.save(to: file)
        fwrite(pointer, byteCount, 1, file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        let length = Int.load(from: file) as! Int
        let capacity = Int.load(from: file) as! Int
        let vec = LLVector<T>.init(capacity: capacity)
        vec.length = length
        fread(vec.pointer, vec.byteCount, 1, file)
        return vec
    }
}

extension Array: Storagable where Element: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        self.count.save(to: file)
        for item in self {
            item.save(to: file)
        }
    }

    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        let count = Int.load(from: file) as! Int
        var arr = [Element]()
        for _ in 0..<count {
            let item: Element = Element.load(from: file) as! Element
            arr.append(item)
        }
        return arr
    }
}

extension NNArray: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        d.save(to: file)
        acci.save(to: file)
        data.save(to: file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        let d = [Int].load(from: file) as! [Int]
        let acci = [Int].load(from: file) as! [Int]
        let arr = NNArray()
        arr.d = d
        arr.acci = acci
        arr.data = LLVector<Float>.load(from: file) as? LLVector<Float>
        return arr
    }
}

extension Matrix: Storagable {
    func save(to file: UnsafeMutablePointer<FILE>) {
        _data.save(to: file)
    }
    
    static func load(from file: UnsafeMutablePointer<FILE>) -> Storagable {
        let arr = NNArray.load(from: file) as! NNArray
        return Matrix(arr)
    }
}
