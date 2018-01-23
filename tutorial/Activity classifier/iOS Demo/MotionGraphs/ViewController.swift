//
//  ViewController.swift
//  MotionGraphs
//
//  Created by Robin on 22/01/2018.
//  Copyright © 2018 TendCloud. All rights reserved.
//

import UIKit
import CoreML
import CoreMotion

class ViewController: UIViewController {
    
    
    @IBOutlet weak var statusLabel: UILabel!
    
    let activityClassificationModel = MyActivityClassifier()
    
    var currentIndexInPredictionWindow = 0
    let predictionWindowDataArray = try? MLMultiArray(
        shape: [1, ModelConstants.predictionWindowSize, ModelConstants.numOfFeatures] as [NSNumber],
        dataType: MLMultiArrayDataType.double)
    var lastHiddenOutput = try? MLMultiArray(
        shape: [ModelConstants.hiddenInLength as NSNumber],
        dataType: MLMultiArrayDataType.double)
    var lastHiddenCellOutput = try? MLMultiArray(
        shape: [ModelConstants.hiddenCellInLength as NSNumber],
        dataType: MLMultiArrayDataType.double)
    
    
    let motionManager: CMMotionManager? = CMMotionManager()
    

    func startMotionSensor() {
        guard let motionManager = motionManager, motionManager.isAccelerometerAvailable && motionManager.isGyroAvailable else { return }
        motionManager.accelerometerUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)
        motionManager.gyroUpdateInterval = TimeInterval(ModelConstants.sensorsUpdateInterval)
        
        // Accelerometer sensor
        motionManager.startAccelerometerUpdates()
        motionManager.startGyroUpdates()
//        motionManager.startAccelerometerUpdates(to: .main) { (accelerometerData, error) in
//            guard let accelerometerData = accelerometerData else {return}
//
//            // add the current acc data sample to the data array
//            self.addSampleToDataArray(accelerometerSample: accelerometerData, gyroSample: nil)
//        }
//        // Gyro sensor
//        motionManager.startGyroUpdates(to: .main) { (gyroData, error) in
//            guard let gyroData = gyroData else { return }
//
//            // add the current gyro data sample to the data array
//            self.addSampleToDataArray(accelerometerSample: nil, gyroSample: gyroData);
//        }
        
        
        Timer.scheduledTimer(withTimeInterval: TimeInterval(ModelConstants.sensorsUpdateInterval), repeats: true) { (timer) in
            self.getSensorSamples()
        }
        
     }
    
    func getSensorSamples() {
        
        guard let motionManager = motionManager else { return }
        
        guard let dataArray = predictionWindowDataArray  else {
            return
        }
        
        let accelerometerSample: CMAccelerometerData = motionManager.accelerometerData!
        let gyroSample: CMGyroData = motionManager.gyroData!
        
        dataArray[[0, currentIndexInPredictionWindow, 0] as [NSNumber]] = accelerometerSample.acceleration.x as NSNumber
        dataArray[[0, currentIndexInPredictionWindow, 1] as [NSNumber]] = accelerometerSample.acceleration.y as NSNumber
        dataArray[[0, currentIndexInPredictionWindow, 2] as [NSNumber]] = accelerometerSample.acceleration.z as NSNumber
        dataArray[[0, currentIndexInPredictionWindow, 3] as [NSNumber]] = gyroSample.rotationRate.x as NSNumber
        dataArray[[0, currentIndexInPredictionWindow, 4] as [NSNumber]] = gyroSample.rotationRate.y as NSNumber
        dataArray[[0, currentIndexInPredictionWindow, 5] as [NSNumber]] = gyroSample.rotationRate.z as NSNumber
        
        // update the index in the prediction window data array
        currentIndexInPredictionWindow += 1
        
        // If the data array is full, call the prediction method to get a new model prediction.
        // We assume here for simplicity that the Gyro data was added to the data array as well.
        if (currentIndexInPredictionWindow == ModelConstants.predictionWindowSize) {
            // predict activity
            let predictedActivity = performModelPrediction() ?? "N/A"
            
            // user the predicted activity here
            print("Current activity: \(predictedActivity)")
            self.statusLabel.text = predictedActivity
            // start a new prediction window
            currentIndexInPredictionWindow = 0
        }
    }
    
    func performModelPrediction () -> String?{
        guard let dataArray = predictionWindowDataArray else { return "Error!"}
        
        // perform model prediction
        let modelPrediction = try? activityClassificationModel.prediction(features: dataArray, hiddenIn: lastHiddenOutput, cellIn: lastHiddenCellOutput)
        
        // update the state vectors
        lastHiddenOutput = modelPrediction?.hiddenOut
        lastHiddenCellOutput = modelPrediction?.cellOut
        
        // return the predicted activity -- the activity with the highest probability
        return modelPrediction?.activity
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        startMotionSensor()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

