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
            fatalError("cannot read file \(path)")
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
//            print("property：\(name)")
            
            let val = children.first(where: { $0.label == name })!.value
            if let value = val as? Storagable {
//                print(value)
                value.save(to: file)
            } else if let layers = val as? [Layer] {
//                print(layers)
                for layer in layers {
//                    print(layer)
                    layer.save(to: file)
                }
            } else if let layer = val as? Layer {
//                print(layer)
                layer.save(to: file)
            } else if let containers = val as? [Container] {
//                print(containers)
                for container in containers {
//                    print(container)
                    container.save(to: file)
                }
            } else if let container = val as? Container {
//                print(container)
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
//            print("property：\(name)")
            
            let value = children.first(where: { $0.label == name })!.value
            if let val = value as? Storagable {
//                print(val)
                c.setValue(type(of: val).load(from: file), forKey: name)
            } else if let container = value as? Container {
//                print(container)
                container.load(from: file)
            } else if let containers = value as? [Container] {
//                print(containers)
                for i in 0..<containers.count {
                    print(containers[i])
                    containers[i].load(from: file)
                }
            } else if let layers = value as? [Layer] {
//                print(layers)
                for i in 0..<layers.count {
//                    print(layers[i])
                    layers[i].load(from: file)
                }
            } else if let layer = value as? Layer {
//                print(layer)
                layer.load(from: file)
            }
        }

        free(plist)
    }
}
