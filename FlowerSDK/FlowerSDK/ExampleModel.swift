//
//  ExampleModel.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 17.03.22.
//

import Foundation
import CoreML
import Vision
import SwiftUI

class ExampleModel: ObservableObject {
    private var coreMLClient: CoreMLClient?
    var text: String = "Flower iOS SDK App"
    var imageLabelDictionary: [UIImage : String] = [:]
    
    init() {
        imageLabelDictionary[UIImage(named: "dog1")!] = "Dog"
        imageLabelDictionary[UIImage(named: "dog2")!] = "Dog"
        imageLabelDictionary[UIImage(named: "dog3")!] = "Dog"
        imageLabelDictionary[UIImage(named: "dog4")!] = "Dog"
        imageLabelDictionary[UIImage(named: "cat1")!] = "Cat"
        imageLabelDictionary[UIImage(named: "cat2")!] = "Cat"
        imageLabelDictionary[UIImage(named: "cat3")!] = "Cat"
    }
    
    private func initClient() {
        if coreMLClient == nil {
            let url = Bundle.main.url(forResource: "CatDogUpdatable", withExtension: "mlmodel")
            self.coreMLClient = CoreMLClient(modelUrl: url!, imageLabelDictionary: imageLabelDictionary)
        }
    }
    
    func federatedTraining() {
        initClient()
        startClient(serverHost: "localhost", serverPort: 8080, client: coreMLClient!)
        //startClient(serverHost: "131.159.194.137", serverPort: 8080, client: coreMLClient!)
    }
    
    func localTraining() {
        initClient()
        let result = coreMLClient?.train(modelConfig: MLModelConfiguration())
        print(result)
    }
}
