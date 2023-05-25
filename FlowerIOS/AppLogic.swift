//
//  AppLogic.swift
//  FlowerIOS
//
//  Created by Daniel Nugraha on 24.05.23.
//

import Foundation
import CoreML
import flwr

public class AppLogic: ObservableObject {
    @Published var hostname: String = ""
    @Published var port: Int = 8080
    
    public func startFlowerClient() {
        // spawn new thread to not block the UI thread
        DispatchQueue.global(qos: .default).async {
            // prepare train dataset
            let trainBatchProvider = DataLoader.trainBatchProvider() { _ in }
            
            // prepare test dataset
            let testBatchProvider = DataLoader.testBatchProvider() { _ in }
            
            // load them together
            let dataLoader = MLDataLoader(trainBatchProvider: trainBatchProvider, testBatchProvider: testBatchProvider)
            
            // getting the mlmodel from the resource
            if let url = Bundle.main.url(forResource: "MNIST_Model", withExtension: "mlmodel") {
                do {
                    // compile mlmodel
                    let compiledModelUrl = try MLModel.compileModel(at: url)
                    
                    // inspect the model to be able to access the model parameters
                    // to access the model we need to know the layer name
                    // since the model parameters are stored as key value pairs
                    let modelInspect = try MLModelInspect(serializedData: Data(contentsOf: url))
                    let layerWrappers = modelInspect.getLayerWrappers()
                    
                    // instantiate the flower client with CoreML
                    let mlFlwrClient = MLFlwrClient(layerWrappers: layerWrappers,
                                                    dataLoader: dataLoader,
                                                    compiledModelUrl: compiledModelUrl)
                    
                    // instantiate grpc client and start federated learning
                    let flwrGRPC = FlwrGRPC(serverHost: self.hostname, serverPort: self.port)
                    flwrGRPC.startFlwrGRPC(client: mlFlwrClient)
                } catch {
                    print(error)
                }
            }
        }
    }
}
