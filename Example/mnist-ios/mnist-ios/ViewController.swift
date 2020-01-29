//
//  ViewController.swift
//  mnist-ios
//
//  Created by Liuliet.Lee on 29/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import UIKit

class ViewController: UIViewController {
    
    @IBOutlet weak var drawView: DrawView!
    @IBOutlet weak var resButton: UIButton!
    
    let reader = ImageReader()

    let model = Sequential([
        Conv(3, count: 3, padding: 1),
        Conv(3, count: 3, padding: 1),
        Conv(3, count: 3, padding: 1),
        ReLU(),
        MaxPool(2, step: 2),
        Conv(3, count: 6, padding: 1),
        Conv(3, count: 6, padding: 1),
        Conv(3, count: 6, padding: 1),
        ReLU(),
        MaxPool(2, step: 2),
        Dense(inFeatures: 6 * 7 * 7, outFeatures: 120),
        Dense(inFeatures: 120, outFeatures: 10)
    ])
    
    override func viewDidLoad() {
        super.viewDidLoad()
        let path = Bundle.main.path(forResource: "mnistmodel01", ofType: "nnm")!
        ModelStorage.load(model, path: path)
        Core.device = MTLCreateSystemDefaultDevice()
    }

    @IBAction func check() {
        let img = getViewImage()
        let res = model.forward(img)
        let pred = res.indexOfMax()
        resButton.setTitle("\(pred)", for: .normal)
    }
    
    func getViewImage() -> NNArray {
        return reader.readCIImage(
            drawView.asImage().resized().asCIImage()!
        ).subArray(pos: 0, length: 784, d: [1, 1, 28, 28])
    }
    
    @IBAction func clear() {
        drawView.clear()
    }
}

extension UIView {
    func asImage() -> UIImage {
        let renderer = UIGraphicsImageRenderer(bounds: bounds)
        return renderer.image { rendererContext in
            layer.render(in: rendererContext.cgContext)
        }
    }
}

extension UIImage {
    func resized(to size: CGSize = CGSize(width: 28, height: 28)) -> UIImage {
        UIGraphicsBeginImageContext(size)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!
        return resizedImage
    }
    
    func asCIImage() -> CIImage? {
        return CIImage(image: self)
    }
}
