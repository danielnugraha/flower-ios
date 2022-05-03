//
//  FlowerSDKApp.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 17.12.21.
//

import SwiftUI

@main
struct FlowerSDKApp: App {
    @StateObject var model = ExampleModel()
    
    var body: some Scene {
        WindowGroup {
            Text(model.text)
                .font(.largeTitle)
            Button("Federated Training") {
                model.federatedTraining()
            }
            .tint(.blue)
            .buttonStyle(.bordered)
            .padding()
            Button("Local Training") {
                model.localTraining()
            }
            .tint(.blue)
            .buttonStyle(.bordered)
        }
    }
}
