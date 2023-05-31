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
    @Published var taskStatus: TaskStatus = .idle
    
    public func startFlowerClient() {
        taskStatus = .preparing(info: "Preparing flower client")
        
        // spawn new thread to not block the UI thread
        DispatchQueue.global(qos: .default).async {
            
            // prepare train dataset
            let trainBatchProvider = DataLoader.trainBatchProvider() { count in
                DispatchQueue.main.async {
                    self.taskStatus = .preparing(info: "Preparing train dataset: \(count)")
                }
            }
            
            // prepare test dataset
            let testBatchProvider = DataLoader.testBatchProvider() { count in
                DispatchQueue.main.async {
                    self.taskStatus = .preparing(info: "Preparing test dataset: \(count)")
                }
            }
            
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
                    DispatchQueue.main.async {
                        self.taskStatus = .ongoing(info: "Federated learning started")
                    }
                    flwrGRPC.startFlwrGRPC(client: mlFlwrClient) {
                        DispatchQueue.main.async {
                            self.taskStatus = .completed(info: "Federated learning completed")
                        }
                    }
                } catch {
                    print(error)
                }
            }
        }
    }
}

public enum TaskStatus: Equatable {
    case idle
    case preparing(info: String)
    case ongoing(info: String)
    case completed(info: String)
    
    var description: String {
        switch self {
        case .idle:
            return "Start Federated Learning"
        case .ongoing(let info):
            return info
        case .completed(let info):
            return info
        case .preparing(info: let info):
            return info
        }
    }
    
    public static func ==(lhs: TaskStatus, rhs: TaskStatus) -> Bool {
        switch(lhs, rhs) {
        case (.idle, .idle):
            return true
        case (.ongoing(_), .ongoing(_)):
            return true
        case (.completed(_), .completed(_)):
            return true
        case (.preparing(_), .preparing(_)):
            return true
        default:
            return false
        }
    }
}
