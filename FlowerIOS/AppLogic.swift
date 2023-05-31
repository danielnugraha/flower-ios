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
