//
//  CoreMLClient.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 04.03.22.
//

import Foundation
import Vision
import CoreML
import SwiftUI

class CoreMLClient: Client {
    
    var model : MLModel?
    
    private var parameterConverter = ParameterConverter()
    
    var layerWrappers: [MLLayerWrapper] = []
    
    var imageLabelDictionary: [UIImage : String]
    
    /// The location of the app's Application Support directory for the user.
    private static let appDirectory = FileManager.default.urls(for: .applicationSupportDirectory,
                                                               in: .userDomainMask).first!
    /// The default model's file URL.
    private let defaultModelURL: URL
    /// The permanent location of the updated model.
    private var updatedModelURL = appDirectory.appendingPathComponent("CatDogUpdatable.mlmodelc")
    /// The temporary location of the updated model.
    private var tempUpdatedModelURL = appDirectory.appendingPathComponent("CatDogUpdatable_tmp.mlmodelc")
    
    init(modelUrl url: URL, imageLabelDictionary: [UIImage : String]) {
        let compiledModelUrl = try! MLModel.compileModel(at: url)
        self.defaultModelURL = compiledModelUrl
        self.updatedModelURL = compiledModelUrl
        self.imageLabelDictionary = imageLabelDictionary
        self.model = loadModel(url: compiledModelUrl)
        if let modelInspect = try? MLModelInspect(serializedData: Data(contentsOf: url)) {
            self.layerWrappers = modelInspect.getLayerWrappers()
        }
        
    }
    
    private func convertToDataTensor(data: Any?) -> Data? {
        try? JSONSerialization.data(withJSONObject: data as Any)
    }
    
    func getParameters() -> ParametersRes {
        print(layerWrappers.map{ $0.shape })
        let parameters = weightsToParameters()
        return ParametersRes(parameters: parameters)
    }
    
    func fit(ins: FitIns) -> FitRes {
        let result = train(modelConfig: parametersToWeights(parameters: ins.parameters))
        let parameters = weightsToParameters()
        return FitRes(parameters: parameters, numExamples: result.numSamples)
    }
    
    func evaluate(ins: EvaluateIns) -> EvaluateRes {
        let result = test(modelConfig: parametersToWeights(parameters: ins.parameters))
        return EvaluateRes(loss: Float(result.loss), numExamples: result.numSamples)
    }
    
    func parametersToWeights(parameters: Parameters) -> MLModelConfiguration {
        parameterConverter = ParameterConverter()
        let config = MLModelConfiguration()
        
        guard parameters.tensors.count == self.layerWrappers.count else {
            print("parameters received is not valid")
            return config
        }
        
        for (index, data) in parameters.tensors.enumerated() {
            print(layerWrappers[index].shape)
            let expectedNumberOfElements = layerWrappers[index].shape.map({Int($0)}).reduce(1, *)
            print(expectedNumberOfElements)
            if let weightsArray = parameterConverter.dataToArray(data: data) {
                guard weightsArray.count == expectedNumberOfElements else {
                    print("array received has wrong number of elements")
                    continue
                }
                
                layerWrappers[index].weights = weightsArray
                
                if layerWrappers[index].isUpdatable {
                    if let weightsMultiArray = parameterConverter.dataToMultiArray(data: data) {
                        let weightsShape = weightsMultiArray.shape.map({ Int16(truncating: $0) }).filter { $0 > 1 }
                        guard weightsShape == layerWrappers[index].shape else {
                            print("shape not the same")
                            continue
                        }
                        let paramKey = MLParameterKey.weights.scoped(to: layerWrappers[index].name)
                        config.parameters?[paramKey] = weightsMultiArray
                    }
                }
            }
        }
        
        return config
    }
    
    func weightsToParameters() -> Parameters {
        parameterConverter = ParameterConverter()
        let dataArray = layerWrappers.compactMap { parameterConverter.arrayToData(array: $0.weights, shape: $0.shape) }
        if dataArray.count != layerWrappers.count {
            print("dataArray size != layerWrappers size")
        }
        return Parameters(tensors: dataArray, tensorType: "ndarray")
    }
    
    func train(modelConfig: MLModelConfiguration) -> MLResult {
        /// The URL of the currently active Model
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL
        
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: currentModelURL.path) {
            print(currentModelURL)
        }
        
        var loss = 0.0
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            /// check progress after epoch
            progressHandler: { (contextProgress) in
                print("param \(contextProgress.parameters)")
                print(contextProgress.metrics)
                print(contextProgress.metrics[.lossValue])
                loss = contextProgress.metrics[.lossValue] as! Double
            
         }) { (finalContext) in
             
             for (index, layer) in self.layerWrappers.enumerated() {
                 print("checking if layer \(layer.name) updatable")
                 if layer.isUpdatable {
                     let paramKey = MLParameterKey.weights.scoped(to: layer.name)
                     print("retrieving parameters for layer \(layer.name)")
                     if let weightsMultiArray = try? finalContext.model.parameterValue(for: paramKey) as? MLMultiArray {
                         let weightsShape = weightsMultiArray.shape.map({ Int16(truncating: $0) }).filter({ $0 > 1 })
                         guard weightsShape == layer.shape else {
                             print("shape is not the same")
                             continue
                         }
                         
                         if let pointer = try? UnsafeBufferPointer<Float>(weightsMultiArray) {
                             print("updating weights for layer \(layer.name)")
                             let array = pointer.compactMap{$0}
                             print("layer weight array size = \(layer.weights.count)")
                             print("array size = \(array.count)")
                             self.layerWrappers[index].weights = array
                         }
                         
                     }
                     
                 }
             }
             
             self.saveModel(finalContext)
             
             self.model = self.loadModel(url: self.updatedModelURL)
         }
        let trainingData = batchProvider(imageLabelDictionary: self.imageLabelDictionary)
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                              trainingData: trainingData,
                                              configuration: modelConfig,
                                              progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            print(error)
        }
        let result = MLResult(loss: loss, numSamples: trainingData.count, accuracy: loss / Double(trainingData.count))
        return result
    }
    
    func test(modelConfig: MLModelConfiguration) -> MLResult {
        /// The URL of the currently active Model
        let usingUpdatedModel = model != nil
        let currentModelURL = usingUpdatedModel ? updatedModelURL : defaultModelURL

        var loss = 0.0
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: [.epochEnd],
            /// check progress after epoch
            progressHandler: { (contextProgress) in
                print(contextProgress.metrics)
                loss = contextProgress.metrics[.lossValue] as! Double
                
            }
        ) { _ in }
        
        let trainingData = batchProvider(imageLabelDictionary: self.imageLabelDictionary)
        do {
            let updateTask = try MLUpdateTask(forModelAt: currentModelURL,
                                          trainingData: trainingData,
                                          configuration: modelConfig,
                                          progressHandlers: progressHandlers)
            updateTask.resume()
        } catch {
            
        }
        let result = MLResult(loss: loss, numSamples: trainingData.count, accuracy: loss / Double(trainingData.count))
        return result
    }
    
    // MARK: - Private Type Helper Methods
    /// Saves the model in the given Update Context provided by an MLUpdateTask.
    /// - Parameter updateContext: The context from the Update Task that contains the updated model.
    /// - Tag: SaveModel
    private func saveModel(_ updateContext: MLUpdateContext) {
        let updatedModel = updateContext.model
        let fileManager = FileManager.default
        do {
            // Create a directory for the updated model.
            try fileManager.createDirectory(at: tempUpdatedModelURL,
                                            withIntermediateDirectories: true,
                                            attributes: nil)
            
            // Save the updated model to temporary filename.
            try updatedModel.write(to: tempUpdatedModelURL)
            
            // Replace any previously updated model with this one.
            _ = try fileManager.replaceItemAt(updatedModelURL,
                                              withItemAt: tempUpdatedModelURL)
            
            print("Updated model saved to:\n\t\(updatedModelURL)")
        } catch let error {
            print("Could not save updated model to the file system: \(error)")
            return
        }
    }
    
    /// Load Model if available.
    /// - Tag: LoadModel
    private func loadModel(url: URL) -> MLModel? {
        guard FileManager.default.fileExists(atPath: url.path) else {
            // The updated model is not present at its designated path.
            return nil
        }
        do {
            // Create an instance of the model.
            return try MLModel(contentsOf: url)
        } catch {
            return nil
        }
    }
    
    private func batchProvider(imageLabelDictionary: [UIImage : String]) -> MLArrayBatchProvider {
        var batchInputs: [MLFeatureProvider] = []
        let imageConstraint = model!.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!
        let imageOptions: [MLFeatureValue.ImageOption: Any] = [
          .cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue
        ]
        for (image,label) in imageLabelDictionary {
            
            do{
                let featureValue = try MLFeatureValue(cgImage: image.cgImage!, constraint: imageConstraint, options: imageOptions)
              
                if let pixelBuffer = featureValue.imageBufferValue{
                    let x = CatDogUpdatableTrainingInput(image: pixelBuffer, classLabel: label)
                    batchInputs.append(x)
                }
            }
            catch(let error){
                print("error description is \(error.localizedDescription)")
            }
        }
     return MLArrayBatchProvider(array: batchInputs)
    }
}
