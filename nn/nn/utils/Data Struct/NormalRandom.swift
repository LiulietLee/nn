//
//  NormalRandom.swift
//  nn
//
//  Created by Liuliet.Lee on 4/1/2020.
//  Copyright © 2020 Liuliet.Lee. All rights reserved.
//

import Foundation

extension NNArray {
    @discardableResult
    public func normalRandn(sigma: Float = 1.0, mu: Float = 0.0) -> NNArray {
        let len = count - count % 2
        
        for i in stride(from: 0, to: len, by: 2) {
            var x = Float(0.0), y = Float(0.0), rsq = Float(0.0), f = Float(0.0)
            repeat {
                x = 2.0 * Float.random(in: 0.0..<1.0) - 1.0
                y = 2.0 * Float.random(in: 0.0..<1.0) - 1.0
                rsq = x * x + y * y
            } while rsq >= 1.0 || rsq == 0.0
            f = sqrt(-2.0 * log(rsq) / rsq)
            self[i] = sigma * (x * f) + mu
            self[i + 1] = sigma * (y * f) + mu
        }
        
//       int i;
//       int m = n + n % 2;
//       double* values = (double*)calloc(m,sizeof(double));
//       double average, deviation;
    
//       if ( values )
//       {
//           for ( i = 0; i < m; i += 2 )
//           {
//               double x,y,rsq,f;
//               do {
//                   x = 2.0 * rand() / (double)RAND_MAX - 1.0;
//                   y = 2.0 * rand() / (double)RAND_MAX - 1.0;
//                   rsq = x * x + y * y;
//               }while( rsq >= 1. || rsq == 0. );
//               f = sqrt( -2.0 * log(rsq) / rsq );
//               values[i]   = x * f;
//               values[i+1] = y * f;
//           }
//       }

        return self
    }

}
