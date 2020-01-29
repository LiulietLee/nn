# nn
> A toy convolutional neural network framework, written by Swift and Metal, for iOS and macOS devices.

## Usage

### Train a handwritten digit classifier

### On [macOS]()

1. Donwload MNIST dataset
```
curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip t*-ubyte.gz
```

2. Define a network
```swift
let net = Sequential([
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
```

3. Create a data reader
```swift
let reader = MNISTReader(
    root: "/Users/.../mnist",
    batchSize: 16
)
```

4. Use GPU (or the computation will be extremely slow)
```swift
Core.device = MTLCreateSystemDefaultDevice()
```

5. Train
```swift
func train() {
    for i in 0..<10 {
        var j = 0
        var runningLoss: Float = 0.0
        while let (img, label) = reader.nextTrain() {
            net.zeroGrad()
            let _ = net.forward(img)
            let loss = net.loss(label)
            net.backward(label)
            net.step(lr: 0.0001, momentum: 0.99)
            runningLoss = max(runningLoss, loss)
            
            if j % 100 == 99 {
                print("[\(i), \(j)] loss: \(runningLoss)")
                runningLoss = 0.0
                ModelStorage.save(net, path: "mnistmodel01.nnm")
            }
            
            j += 1
        }
    }
    
    ModelStorage.save(net, path: "mnistmodel01.nnm")
}

train()
```
Each epoch takes about 20 minutes on my computer (MacBook Pro Retina, 13-inch, Mid 2014)

Maybe you think that it's very slow.

Yes. So it's recommended to start training before you go to bed.

6. Test
```swift
func test() {
    ModelStorage.load(net, path: "mnistmodel01.nnm")
    
    var cor = 0
    var tot = 0
    while let (img, label) = reader.nextTest() {
        let score = net.forward(img)
        let pred = score.indexOfMax()
        if pred == label {
            cor += 1
            print("\(tot): Y \(pred) == \(label)")
        } else {
            print("\(tot): N \(pred) != \(label)")
        }
        tot += 1
    }
    print("correct: \(cor) / \(tot)")
}

test()
```
The accuracy rate can reach at least 96%.

### On [iOS]()

1. Move pretrained model [mnistmodel01.nnm]() file to your iOS project. Make sure the **Target Membership** is selected.

2. Create a model with the same structure as before.
```swift
let model = Sequential([
    Conv(3, count: 3, padding: 1),
    ...
    Dense(inFeatures: 120, outFeatures: 10)
])
```

3. Load model parameters and enable GPU computation in `viewDidLoad()`.
```swift
override func viewDidLoad() {
    super.viewDidLoad()
    let path = Bundle.main.path(forResource: "mnistmodel01", ofType: "nnm")!
    ModelStorage.load(model, path: path)
    Core.device = MTLCreateSystemDefaultDevice()
}
```

4. Use `forward()` function to predict.
```swift
@IBAction func check() {
    let img = getViewImage() // this function convert UIView to NNArray
    let res = model.forward(img)
    let pred = res.indexOfMax()
    button.setTitle("\(pred)", for: .normal)
}
```
You can read the [source]() for detail.

## Avaliable Layer
- Conv (2D convolutional layer)
- Dense (fully connected layer)
- ReLU (leaky relu)
- MaxPool (2D max pooling layer)
- AveragePool (2D average pooling layer (uncompleted ver.))
