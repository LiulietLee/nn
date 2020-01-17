//
//  ModelStorage.swift
//  nn
//
//  Created by Liuliet.Lee on 30/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class ModelStorage {
    
    public static func save(_ model: Container, path: String) {
        guard let file = fopen(path, "w") else {
            fatalError("cannot write file \(path)")
        }
        model.save(to: file)
        fclose(file)
    }
    
    public static func load(_ model: Container, path: String) {
        guard let file = fopen(path, "r") else {
            print("cannot read file \(path)")
            return
        }
        model.load(from: file)
        fclose(file)
    }
}

extension ModelStorage {
    public static func save(_ c: NSObject, file: UnsafeMutablePointer<FILE>) {
        var count: UInt32 = 0
        let cls: AnyClass = object_getClass(c)!
        let plist = class_copyPropertyList(cls, &count)
        let children = Mirror(reflecting: c).children

        for i in 0..<count {
            let property = plist?[Int(i)]
            let cname = property_getName(property!)
            let name = String(cString: cname)
            
            let val = children.first(where: { $0.label == name })!.value
            if let value = val as? Storagable {
                value.save(to: file)
            } else if let layers = val as? [Layer] {
                for layer in layers {
                    layer.save(to: file)
                }
            } else if let layer = val as? Layer {
                layer.save(to: file)
            } else if let containers = val as? [Container] {
                for container in containers {
                    container.save(to: file)
                }
            } else if let container = val as? Container {
                container.save(to: file)
            }
        }

        free(plist)
    }
    
    public static func load(_ c: NSObject, file: UnsafeMutablePointer<FILE>) {
        var count: UInt32 = 0
        let cls: AnyClass = object_getClass(c)!
        guard let plist = class_copyPropertyList(cls, &count) else { return }
        let children = Mirror(reflecting: c).children

        for i in 0..<count {
            let property = plist[Int(i)]
            let cname = property_getName(property)
            let name = String(cString: cname)
            
            let value = children.first(where: { $0.label == name })!.value
            if let val = value as? Storagable {
                c.setValue(type(of: val).load(from: file), forKey: name)
            } else if let container = value as? Container {
                container.load(from: file)
            } else if let containers = value as? [Container] {
                for i in 0..<containers.count {
                    containers[i].load(from: file)
                }
            } else if let layers = value as? [Layer] {
                for i in 0..<layers.count {
                    layers[i].load(from: file)
                }
            } else if let layer = value as? Layer {
                layer.load(from: file)
            }
        }

        free(plist)
    }
}
