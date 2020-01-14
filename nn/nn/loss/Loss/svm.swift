//
//  svm.swift
//  nn
//
//  Created by Liuliet.Lee on 14/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

extension Loss {
    public class svm: AbstractLoss {
        
        public static var d: Float = 1.0
        
        static func batchLoss(score: NNArray, label: NNArray) -> Float {
            let maxi = label.indexOfMax()
            
            var loss: Float = 0.0
            for i in 0..<score.count {
                if i != maxi {
                    loss += max(0, score[i] - score[maxi] + d)
                }
            }
            
            return loss
        }
        
        public static func loss(score: NNArray, label: NNArray) -> Float {
            var loss: Float = 0.0
            for i in 0..<score.d[0] {
                loss += batchLoss(score: score.subArray(at: i), label: label.subArray(at: i))
            }
            return loss / Float(score.d[0])
        }
        
        static func batchDelta(score: NNArray, label: NNArray) -> NNArray {
            let da = NNArray(score.d)
            
            let maxi = label.indexOfMax()
            
            for i in 0..<label.count {
                if i != maxi {
                    da[i] = score[i] - score[maxi] + d > 0 ? 1 : 0
                } else {
                    for j in 0..<label.count {
                        if j != maxi {
                            da[i] += score[j] - score[maxi] + d > 0 ? -1 : 0
                        }
                    }
                }
            }
            
            return da
        }
        
        public static func delta(score: NNArray, label: NNArray) -> NNArray {
            var res = [NNArray]()
            for i in 0..<score.d[0] {
                res.append(
                    batchDelta(score: score.subArray(at: i), label: label.subArray(at: i))
                )
            }
            return NNArray.concat(res).dim(score.d)
        }
    }
}
