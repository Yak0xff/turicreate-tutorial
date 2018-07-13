//
//  ViewController.swift
//  Vision+ML
//
//  Created by Robin on 2018/7/13.
//  Copyright © 2018 TendCloud. All rights reserved.
//

import UIKit
import CoreML
import Vision
import ImageIO

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var classificationLabel: UILabel!
    
    @IBAction func takePicture(_ sender: Any){
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        present(picker, animated: true)
    }
    
    @IBAction func chooseImage(_ sender: Any){
        // The photo library is the default source, editing not allowed
        let picker = UIImagePickerController()
        picker.delegate = self
        picker.sourceType = .savedPhotosAlbum
        present(picker, animated: true)
    }
    
    var inputImage: CIImage!// The image to be processed.
    
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String : Any]) {
        picker.dismiss(animated: true)
        classificationLabel.text = "Analyzing Image..."
        
        guard let uiImage = info[UIImagePickerControllerOriginalImage] as? UIImage else {
            fatalError("No image from image picker")
        }
        guard let ciImage = CIImage(image: uiImage) else {
            fatalError("Can't create CIImage from UIImage")
        }
        let orientation = CGImagePropertyOrientation(uiImage.imageOrientation)
        inputImage = ciImage.oriented(forExifOrientation: Int32(orientation.rawValue))
        
        // Show the image in the UI.
        imageView.image = uiImage
        
        
        // Run the rectangle detector, which upon completion runs the ML classifier.
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: CGImagePropertyOrientation(rawValue: UInt32(Int32(orientation.rawValue)))!)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([self.rectanglesRequest])
            } catch {
                print(error)
            }
        }
    }
    
    lazy var rectanglesRequest: VNDetectRectanglesRequest = {
        return VNDetectRectanglesRequest(completionHandler: self.handleRectangles)
    }()
    func handleRectangles(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNRectangleObservation]
            else { fatalError("unexpected result type from VNDetectRectanglesRequest") }
        guard let detectedRectangle = observations.first else {
            DispatchQueue.main.async {
                self.classificationLabel.text = "No rectangles detected."
            }
            return
        }
        let imageSize = inputImage.extent.size
        
        // Verify detected rectangle is valid.
        let boundingBox = detectedRectangle.boundingBox.scaled(to: imageSize)
        guard inputImage.extent.contains(boundingBox)
            else { print("invalid detected rectangle"); return }
        
        // Rectify the detected image and reduce it to inverted grayscale for applying model.
        let topLeft = detectedRectangle.topLeft.scaled(to: imageSize)
        let topRight = detectedRectangle.topRight.scaled(to: imageSize)
        let bottomLeft = detectedRectangle.bottomLeft.scaled(to: imageSize)
        let bottomRight = detectedRectangle.bottomRight.scaled(to: imageSize)
        let correctedImage = inputImage
            .cropped(to: boundingBox)
            .applyingFilter("CIPerspectiveCorrection", parameters: [
                "inputTopLeft": CIVector(cgPoint: topLeft),
                "inputTopRight": CIVector(cgPoint: topRight),
                "inputBottomLeft": CIVector(cgPoint: bottomLeft),
                "inputBottomRight": CIVector(cgPoint: bottomRight)
                ])
            .applyingFilter("CIColorControls", parameters: [
                kCIInputSaturationKey: 0,
                kCIInputContrastKey: 32
                ])
            .applyingFilter("CIColorInvert", parameters: [:])
        
        
        // Run the Core ML MNIST classifier -- results in handleClassification method
        let handler = VNImageRequestHandler(ciImage: correctedImage)
        do {
            try handler.perform([classificationRequest])
        } catch {
            print(error)
        }
    }
    
    lazy var classificationRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: MyImageClassifier().model)
            return VNCoreMLRequest(model: model, completionHandler: self.handleClassification)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    func handleClassification(request: VNRequest, error: Error?) {
        guard let observations = request.results as? [VNClassificationObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        guard let best = observations.first
            else { fatalError("can't get best result") }
        
        DispatchQueue.main.async {
            self.classificationLabel.text = "Classification: \"\(best.identifier)\" Confidence: \(best.confidence)"
        }
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

