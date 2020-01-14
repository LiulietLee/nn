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

    public static var fileManager = FileManager.default

    public enum ImageType {
        case jpg
        case png
    }
    
    public func readImage(path: String, type: ImageType = .png, buffer: NNArray? = nil, batch: Int = 0) -> NNArray? {
        let url = URL(fileURLWithPath: path)

        if ImageReader.fileManager.fileExists(atPath: url.path),
            let ciimage = CIImage(contentsOf: url) {
            return readCIImage(ciimage, type: type, buffer: buffer, batch: batch)
        } else {
            return nil
        }
    }
    
    public func readCIImage(_ image: CIImage, type: ImageType = .png, buffer: NNArray? = nil, batch: Int = 0) -> NNArray {
        let context = CIContext(options: nil)
        let cgimage = context.createCGImage(image, from: image.extent)!
        
        let imageWidth = cgimage.width
        let imageHeight = cgimage.height
        let arr = buffer == nil ? NNArray(1, 3, imageWidth, imageHeight) : buffer!
        
        let pixelData = cgimage.dataProvider!.data
        let data: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
        let channel = type == .png ? 4 : 3
                
        for i in 0..<imageWidth {
            for j in 0..<imageHeight {
                autoreleasepool {
                    let pixelInfo = channel * ((imageWidth * j) + i)
                    
                    arr[batch, 0, i, j] = Float(data[pixelInfo]) / 255.0
                    arr[batch, 1, i, j] = Float(data[pixelInfo + 1]) / 255.0
                    arr[batch, 2, i, j] = Float(data[pixelInfo + 2]) / 255.0
                }
            }
        }
                
        return arr
    }
}
