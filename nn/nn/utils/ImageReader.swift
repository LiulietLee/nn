//
//  ImageReader.swift
//  nn
//
//  Created by Liuliet.Lee on 20/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation
import CoreImage

public class ImageReader {
    
    public func cifar10Image(path: String) -> NNArray? {
        let url = URL(fileURLWithPath: path)
        let fileManager = FileManager.default

        if fileManager.fileExists(atPath: url.path) {
            let arr = NNArray(32, 32, 3)

            let ciimage = CIImage(contentsOf: url)!
            let context = CIContext(options: nil)
            let cgimage = context.createCGImage(ciimage, from: ciimage.extent)!
            
            let imageWidth = cgimage.width
            let imageHeight = cgimage.height
            
            let pixelData = cgimage.dataProvider!.data
            let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
            
            for i in 0..<imageWidth {
                for j in 0..<imageHeight {
                    let pixelInfo = 4 * ((imageWidth * j) + i)
                    
                    arr[i, j, 0] = Float(data[pixelInfo]) / 255.0
                    arr[i, j, 1] = Float(data[pixelInfo + 1]) / 255.0
                    arr[i, j, 2] = Float(data[pixelInfo + 2]) / 255.0
                }
            }
            
            return arr
        } else {
            return nil
        }
    }
}
