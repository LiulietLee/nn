//
//  MaxPooling.swift
//  nn
//
//  Created by Liuliet.Lee on 19/12/2019.
//  Copyright © 2019 Liuliet.Lee. All rights reserved.
//

import Foundation

public class MaxPool: BaseLayer {
    public struct Position {
        var p = SIMD3<Int32>()
        
        public init() {}
        
        public init(_ x: Int, _ y: Int, _ z: Int) {
            p = SIMD3<Int32>(Int32(x), Int32(y), Int32(z))
        }
    }
    
    public struct SwitchMapper {
        var batch: Int32
        var inPos: Position, outPos: Position
        
        var inputPosition: [Int] {
            [Int(batch), Int(inPos.p[0]), Int(inPos.p[1]), Int(inPos.p[2])]
        }
        
        var outputPosition: [Int] {
            [Int(batch), Int(outPos.p[0]), Int(outPos.p[1]), Int(outPos.p[2])]
        }
        
        init(_ batch: Int, _ inPos: Position, _ outPos: Position) {
            self.batch = Int32(batch)
            self.inPos = inPos
            self.outPos = outPos
        }
    }
    
    public var switches = LLVector<SwitchMapper>()
    var padding = 0
    var width = 0
    var height = 0
    var step = 0
    
    var row = 0
    var col = 0
    
    public init(_ width: Int, _ height: Int = -1, step: Int = 1, padding: Int = 0) {
        self.width = width
        self.height = height <= 0 ? width : height
        self.step = step
        self.padding = padding
    }
    
    private func inBound(_ x: Int, _ y: Int, _ row: Int, _ col: Int) -> Bool {
        return 0 <= x && x < row && 0 <= y && y < col
    }
    
    public override func forward(_ input: NNArray) -> NNArray {
        if batchSize == 0 {
            precondition(
                (input.d[2] - width + padding * 2) % step == 0 &&
                (input.d[3] - height + padding * 2) % step == 0 &&
                padding < width && padding < height
            )
            batchSize = input.d[0]
            row = (input.d[2] - width + padding * 2) / step + 1
            col = (input.d[3] - width + padding * 2) / step + 1
            switches = LLVector<SwitchMapper>(
                capacity: batchSize * input.d[1] * row * col
            )
            switches.length = batchSize * input.d[1] * row * col
            score = NNArray(batchSize, input.d[1], row, col)
        }
        score.data.zero()
        
        if Core.device != nil {
            return forwardWithMetal(input)
        }
        
        for batch in 0..<batchSize {
            for k in 0..<input.d[1] {
                for i in 0..<row {
                    let ri = i * step
                    for j in 0..<col {
                        let rj = j * step
                        var maxPosition = (ri, rj)
                        var maxValue = inBound(ri, rj, input.d[2], input.d[3]) ? input[batch, k, ri, rj] : -Float.infinity
                        for x in 0..<width {
                            for y in 0..<height {
                                let rx = ri + x - padding, ry = rj + y - padding
                                if inBound(rx, ry, input.d[2], input.d[3]),
                                    maxValue < input[batch, k, rx, ry] {
                                    maxValue = input[batch, k, rx, ry]
                                    maxPosition = (rx, ry)
                                }
                            }
                        }
                        switches[batch * input.d[1] * row * col +
                            k * row * col +
                            i * col +
                            j] = SwitchMapper(
                                batch,
                                Position(k, maxPosition.0, maxPosition.1),
                                Position(k, i, j)
                        )
                        score[batch, k, i, j] = maxValue
                    }
                }
            }
        }
        
        return score
    }
    
    public override func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        let da = NNArray(input.count, initValue: 0.0).dim(input.d)
        delta.dim(score.d)
        
        if Core.device != nil {
            return backwardWithMetal(da, input, delta)
        }
        
        for choose in switches {
            let i = choose.inputPosition, j = choose.outputPosition
            da[i[0], i[1], i[2], i[3]] += delta[j[0], j[1], j[2], j[3]]
        }
        
        return da
    }
}
