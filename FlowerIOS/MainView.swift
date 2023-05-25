//
//  MainView.swift
//  FlowerIOS
//
//  Created by Daniel Nugraha on 24.05.23.
//

import SwiftUI

struct MainView: View {
    @ObservedObject var appLogic = AppLogic()
    
    var numberFormatter: NumberFormatter = {
        var nf = NumberFormatter()
        nf.usesGroupingSeparator = false
        nf.numberStyle = .none
        return nf
    }()
    
    var body: some View {
        VStack() {
            Spacer().frame(height: 50)
            Text("Flower iOS Client")
                .font(.largeTitle)
            Spacer().frame(height: 20)
            Form {
                Section(header: Text("Federated Learning")) {
                    HStack {
                        Text("Server Hostname: ")
                        TextField("Server Hostname", text: $appLogic.hostname)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Text("Server Port: ")
                        TextField( "Server Port", value: $appLogic.port, formatter: numberFormatter)
                            .multilineTextAlignment(.trailing)
                    }
                    HStack {
                        Spacer()
                        Button(action: {
                            appLogic.startFlowerClient()
                        }) {
                            Text("Start Federated Learning")
                        }
                    }
                }
            }
        }
        .background(Color(UIColor.systemGray6))
    }
}

struct MainView_Previews: PreviewProvider {
    static var previews: some View {
        MainView()
    }
}
