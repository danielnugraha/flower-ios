//
//  Typing.swift
//  FlowerSDK
//
//  Created by Daniel Nugraha on 09.01.22.
//

import Foundation

public struct Scalar {
    public var bool: Bool?
    public var bytes: Data?
    public var float: Float?
    public var int: Int?
    public var str: String?
}

public typealias Metrics = [String: Scalar]
public typealias Properties = [String: Scalar]

public struct Parameters {
    public var tensors: [Data]
    public var tensorType: String
}

public struct ParametersRes {
    public var parameters: Parameters
}

public struct FitIns {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

public struct FitRes {
    public var parameters: Parameters
    public var numExamples: Int
    public var metrics: Metrics? = nil
}

public struct EvaluateIns {
    public var parameters: Parameters
    public var config: [String: Scalar]
}

public struct EvaluateRes {
    public var loss: Float
    public var numExamples: Int
    public var metrics: Metrics? = nil
}

public struct PropertiesIns {
    public var config: Properties
}

public struct PropertiesRes {
    public var properties: Properties
}

public struct Reconnect {
    public var seconds: Int?
}

public struct Disconnect {
    public var reason: String
}
