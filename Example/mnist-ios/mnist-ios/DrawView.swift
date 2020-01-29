//
//  DrawView.swift
//  ox
//
//  Created by Liuliet.Lee on 17/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import UIKit

class Line {
    var start: CGPoint
    var end: CGPoint
    var color: UIColor
    var width: CGFloat
    
    init(start: CGPoint, end: CGPoint, color: UIColor, width: CGFloat) {
        self.start = start
        self.end = end
        self.color = color
        self.width = width
    }
    
}

class DrawView: UIView {

    var lineWidth = CGFloat(16.0)
    var color = UIColor.white
    
    private var lines = [[Line]]()
    private var trash = [[Line]]()
    private var lastPoint: CGPoint!
    private var isSwipe = false
    
    override func draw(_ rect: CGRect) {
        let context = UIGraphicsGetCurrentContext()
        context?.setLineCap(CGLineCap.round)

        for line in lines {
            for segment in line {
                context?.beginPath()
                
                context?.move(to: CGPoint(x: segment.start.x, y: segment.start.y))
                context?.addLine(to: CGPoint(x: segment.end.x, y: segment.end.y))
                
                context?.setLineWidth(segment.width)
                context?.setStrokeColor(segment.color.cgColor)
                context?.strokePath()
            }
        }
    }
    
    required init?(coder aDecoder: NSCoder) {
        super.init(coder: aDecoder)
    }
    
    func undo() {
        if ((self.viewWithTag(450) as? UIImageView) != nil) {
            while let view = self.viewWithTag(450) as? UIImageView {
                view.removeFromSuperview()
            }
        } else if lines.count != 0 {
            trash.append(lines.last!)
            lines.removeLast()
        }
        self.setNeedsDisplay()
    }
    
    func redo() {
        if trash.count != 0 {
            lines.append(trash.last!)
            trash.removeLast()
        }
        self.setNeedsDisplay()
    }
    
    func clear() {
        while let view = self.viewWithTag(450) as? UIImageView {
            view.removeFromSuperview()
        }
        
        lines = [[Line]]()
        trash = [[Line]]()
        self.setNeedsDisplay()
    }
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with:event)
        
        if !touches.isEmpty {
            lines.append([Line]())
            lastPoint = touches.first!.location(in: self)
            isSwipe = false
            trash = [[Line]]()
        }
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with:event)

        if !touches.isEmpty {
            let newPoint = touches.first!.location(in: self)
            
            lines[lines.count - 1].append(
                Line(start: lastPoint, end: newPoint, color: color, width: lineWidth)
            )
            
            lastPoint = newPoint
            isSwipe = true
            self.setNeedsDisplay()
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with:event)

        if !touches.isEmpty {
            if !isSwipe {
                if let point = lastPoint {
                    lines[lines.count - 1].append(
                        Line(start: point, end: point, color: color, width: lineWidth)
                    )
                }
                self.setNeedsDisplay()
            }
        }
    }
    

}
