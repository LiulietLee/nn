//
//  Inception.swift
//  nn
//
//  Created by Liuliet.Lee on 19/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

public class Inception: BaseLayer {
    
    @objc dynamic var core = [Sequential]()
    var interScore = [NNArray]()
    var interDelta = [NNArray]()
    var input = NNArray()
    var outChannelIndex = [Int]()
    
    var pool = ThreadPool(count: 0)
    
    public override init() {
        super.init()
    }
    
    public init(_ description: [[BaseLayer]]) {
        core = description.map { Sequential($0) }
    }
    
    private func placeholderPrepare() {
        if interScore.count == 0 {
            pool = ThreadPool(count: core.count)
            interScore = [NNArray](repeating: NNArray(), count: core.count)
            interDelta = [NNArray](repeating: NNArray(), count: core.count)
        }
    }
    
    public override func forward(_ input: NNArray) -> NNArray {
        placeholderPrepare()
        self.input = input.copy()
        pool.run { i in
            self.interScore[i] = self.core[i].forward(input)
        }
        score = NNArray.concat(interScore)
        score.d[1] *= score.d[0] / input.d[0]
        score.d[0] = input.d[0]
        return score
    }
    
    public override func predict(_ input: NNArray) -> NNArray {
        placeholderPrepare()
        pool.run { i in
            self.interScore[i] = self.core[i].predict(input)
        }
        score = NNArray.concat(interScore)
        score.d[1] *= score.d[0] / input.d[0]
        score.d[0] = input.d[0]
        return score
    }
    
    @discardableResult
    public override func backward(_ input: NNArray, delta: NNArray) -> NNArray {
        placeholderPrepare()

        if outChannelIndex.isEmpty {
            var idx = 0
            for c in core {
                outChannelIndex.append(idx)
                let pattern = c.score.d
                let outputChannel = pattern[1]
                idx += outputChannel
            }
        }
        
        pool.run { i in
            let idx = self.outChannelIndex[i]
            let pattern = self.core[i].score.d
            let outputChannel = pattern[1]
            var subDeltas = [NNArray]()
            for batch in 0..<delta.d[0] {
                subDeltas.append(
                    delta.subArray(
                        at: batch, idx,
                        length: outputChannel * delta.acci[1],
                        d: [1, outputChannel, delta.d[2], delta.d[3]]
                    )
                )
            }
            let subDelta = NNArray.concat(subDeltas)
            self.interDelta[i] = self.core[i].backward(input, delta: subDelta)
        }
        
        if Core.device != nil {
            return sumInterDelta()
        }

        let da = interDelta[0]
        for i in 1..<interDelta.count {
            for j in 0..<interDelta[i].count {
                da[i] += interDelta[i][j]
            }
        }
        
        return da
    }
    
    public override func zeroGrad() {
        placeholderPrepare()
        pool.run { i in
            self.core[i].zeroGrad()
        }
    }
    
    public override func step(lr: Float, momentum: Float) {
        placeholderPrepare()
        pool.run { i in
            self.core[i].step(lr: lr, momentum: momentum)
        }
    }
    
}
