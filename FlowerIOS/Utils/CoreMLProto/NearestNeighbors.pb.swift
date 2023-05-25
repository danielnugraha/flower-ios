// DO NOT EDIT.
// swift-format-ignore-file
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: NearestNeighbors.proto
//
// For information on using the generated types, please see the documentation:
//   https://github.com/apple/swift-protobuf/

// Copyright (c) 2017, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in LICENSE.txt or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that you are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

///*
/// A k-Nearest-Neighbor classifier
struct CoreML_Specification_KNearestNeighborsClassifier {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///*
  /// The "core" nearest neighbor model attributes.
  var nearestNeighborsIndex: CoreML_Specification_NearestNeighborsIndex {
    get {return _nearestNeighborsIndex ?? CoreML_Specification_NearestNeighborsIndex()}
    set {_nearestNeighborsIndex = newValue}
  }
  /// Returns true if `nearestNeighborsIndex` has been explicitly set.
  var hasNearestNeighborsIndex: Bool {return self._nearestNeighborsIndex != nil}
  /// Clears the value of `nearestNeighborsIndex`. Subsequent reads from it will return its default value.
  mutating func clearNearestNeighborsIndex() {self._nearestNeighborsIndex = nil}

  ///*
  /// Number of neighbors to use for classification.
  var numberOfNeighbors: CoreML_Specification_Int64Parameter {
    get {return _numberOfNeighbors ?? CoreML_Specification_Int64Parameter()}
    set {_numberOfNeighbors = newValue}
  }
  /// Returns true if `numberOfNeighbors` has been explicitly set.
  var hasNumberOfNeighbors: Bool {return self._numberOfNeighbors != nil}
  /// Clears the value of `numberOfNeighbors`. Subsequent reads from it will return its default value.
  mutating func clearNumberOfNeighbors() {self._numberOfNeighbors = nil}

  ///*
  /// Type of labels supported by the model. Currently supports String or Int64
  /// labels.
  var classLabels: CoreML_Specification_KNearestNeighborsClassifier.OneOf_ClassLabels? = nil

  var stringClassLabels: CoreML_Specification_StringVector {
    get {
      if case .stringClassLabels(let v)? = classLabels {return v}
      return CoreML_Specification_StringVector()
    }
    set {classLabels = .stringClassLabels(newValue)}
  }

  var int64ClassLabels: CoreML_Specification_Int64Vector {
    get {
      if case .int64ClassLabels(let v)? = classLabels {return v}
      return CoreML_Specification_Int64Vector()
    }
    set {classLabels = .int64ClassLabels(newValue)}
  }

  ///*
  /// Default value of class label (useful when prediction is called on an empty kNN classifier)
  var defaultClassLabel: CoreML_Specification_KNearestNeighborsClassifier.OneOf_DefaultClassLabel? = nil

  var defaultStringLabel: String {
    get {
      if case .defaultStringLabel(let v)? = defaultClassLabel {return v}
      return String()
    }
    set {defaultClassLabel = .defaultStringLabel(newValue)}
  }

  var defaultInt64Label: Int64 {
    get {
      if case .defaultInt64Label(let v)? = defaultClassLabel {return v}
      return 0
    }
    set {defaultClassLabel = .defaultInt64Label(newValue)}
  }

  ///*
  /// Weighting scheme to be used when computing the majority label of a 
  /// new data point.
  var weightingScheme: CoreML_Specification_KNearestNeighborsClassifier.OneOf_WeightingScheme? = nil

  var uniformWeighting: CoreML_Specification_UniformWeighting {
    get {
      if case .uniformWeighting(let v)? = weightingScheme {return v}
      return CoreML_Specification_UniformWeighting()
    }
    set {weightingScheme = .uniformWeighting(newValue)}
  }

  var inverseDistanceWeighting: CoreML_Specification_InverseDistanceWeighting {
    get {
      if case .inverseDistanceWeighting(let v)? = weightingScheme {return v}
      return CoreML_Specification_InverseDistanceWeighting()
    }
    set {weightingScheme = .inverseDistanceWeighting(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  ///*
  /// Type of labels supported by the model. Currently supports String or Int64
  /// labels.
  enum OneOf_ClassLabels: Equatable {
    case stringClassLabels(CoreML_Specification_StringVector)
    case int64ClassLabels(CoreML_Specification_Int64Vector)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_ClassLabels, rhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_ClassLabels) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.stringClassLabels, .stringClassLabels): return {
        guard case .stringClassLabels(let l) = lhs, case .stringClassLabels(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.int64ClassLabels, .int64ClassLabels): return {
        guard case .int64ClassLabels(let l) = lhs, case .int64ClassLabels(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  ///*
  /// Default value of class label (useful when prediction is called on an empty kNN classifier)
  enum OneOf_DefaultClassLabel: Equatable {
    case defaultStringLabel(String)
    case defaultInt64Label(Int64)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_DefaultClassLabel, rhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_DefaultClassLabel) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.defaultStringLabel, .defaultStringLabel): return {
        guard case .defaultStringLabel(let l) = lhs, case .defaultStringLabel(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.defaultInt64Label, .defaultInt64Label): return {
        guard case .defaultInt64Label(let l) = lhs, case .defaultInt64Label(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  ///*
  /// Weighting scheme to be used when computing the majority label of a 
  /// new data point.
  enum OneOf_WeightingScheme: Equatable {
    case uniformWeighting(CoreML_Specification_UniformWeighting)
    case inverseDistanceWeighting(CoreML_Specification_InverseDistanceWeighting)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_WeightingScheme, rhs: CoreML_Specification_KNearestNeighborsClassifier.OneOf_WeightingScheme) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.uniformWeighting, .uniformWeighting): return {
        guard case .uniformWeighting(let l) = lhs, case .uniformWeighting(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.inverseDistanceWeighting, .inverseDistanceWeighting): return {
        guard case .inverseDistanceWeighting(let l) = lhs, case .inverseDistanceWeighting(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  init() {}

  fileprivate var _nearestNeighborsIndex: CoreML_Specification_NearestNeighborsIndex? = nil
  fileprivate var _numberOfNeighbors: CoreML_Specification_Int64Parameter? = nil
}

///*
/// The "core" attributes of a Nearest Neighbors model.
struct CoreML_Specification_NearestNeighborsIndex {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///* 
  /// Number of dimensions of the input data.
  var numberOfDimensions: Int32 = 0

  ///*
  /// Vector of floating point data that makes up the model. Each data point must have 'numberOfDimensions'
  /// dimensions.
  var floatSamples: [CoreML_Specification_FloatVector] = []

  ///* 
  /// Backing data structure for the Nearest Neighbors Index. Currently supports 
  /// a linear index or a kd-tree index.
  var indexType: CoreML_Specification_NearestNeighborsIndex.OneOf_IndexType? = nil

  var linearIndex: CoreML_Specification_LinearIndex {
    get {
      if case .linearIndex(let v)? = indexType {return v}
      return CoreML_Specification_LinearIndex()
    }
    set {indexType = .linearIndex(newValue)}
  }

  var singleKdTreeIndex: CoreML_Specification_SingleKdTreeIndex {
    get {
      if case .singleKdTreeIndex(let v)? = indexType {return v}
      return CoreML_Specification_SingleKdTreeIndex()
    }
    set {indexType = .singleKdTreeIndex(newValue)}
  }

  ///* 
  /// Distance function to be used to find neighbors. Currently only Squared Euclidean
  /// Distance is supported.
  var distanceFunction: CoreML_Specification_NearestNeighborsIndex.OneOf_DistanceFunction? = nil

  var squaredEuclideanDistance: CoreML_Specification_SquaredEuclideanDistance {
    get {
      if case .squaredEuclideanDistance(let v)? = distanceFunction {return v}
      return CoreML_Specification_SquaredEuclideanDistance()
    }
    set {distanceFunction = .squaredEuclideanDistance(newValue)}
  }

  var unknownFields = SwiftProtobuf.UnknownStorage()

  ///* 
  /// Backing data structure for the Nearest Neighbors Index. Currently supports 
  /// a linear index or a kd-tree index.
  enum OneOf_IndexType: Equatable {
    case linearIndex(CoreML_Specification_LinearIndex)
    case singleKdTreeIndex(CoreML_Specification_SingleKdTreeIndex)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_NearestNeighborsIndex.OneOf_IndexType, rhs: CoreML_Specification_NearestNeighborsIndex.OneOf_IndexType) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.linearIndex, .linearIndex): return {
        guard case .linearIndex(let l) = lhs, case .linearIndex(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      case (.singleKdTreeIndex, .singleKdTreeIndex): return {
        guard case .singleKdTreeIndex(let l) = lhs, case .singleKdTreeIndex(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      default: return false
      }
    }
  #endif
  }

  ///* 
  /// Distance function to be used to find neighbors. Currently only Squared Euclidean
  /// Distance is supported.
  enum OneOf_DistanceFunction: Equatable {
    case squaredEuclideanDistance(CoreML_Specification_SquaredEuclideanDistance)

  #if !swift(>=4.1)
    static func ==(lhs: CoreML_Specification_NearestNeighborsIndex.OneOf_DistanceFunction, rhs: CoreML_Specification_NearestNeighborsIndex.OneOf_DistanceFunction) -> Bool {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch (lhs, rhs) {
      case (.squaredEuclideanDistance, .squaredEuclideanDistance): return {
        guard case .squaredEuclideanDistance(let l) = lhs, case .squaredEuclideanDistance(let r) = rhs else { preconditionFailure() }
        return l == r
      }()
      }
    }
  #endif
  }

  init() {}
}

///*
/// Specifies a uniform weighting scheme (i.e. each neighbor receives equal
/// voting power).
struct CoreML_Specification_UniformWeighting {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// Specifies a inverse-distance weighting scheme (i.e. closest neighbors receives higher
/// voting power). A nearest neighbor with highest sum of (1 / distance) is picked.
struct CoreML_Specification_InverseDistanceWeighting {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// Specifies a flat index of data points to be searched by brute force.
struct CoreML_Specification_LinearIndex {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// Specifies a kd-tree backend for the nearest neighbors model.
struct CoreML_Specification_SingleKdTreeIndex {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  ///*
  /// Number of data points contained within a leaf node of the kd-tree.
  var leafSize: Int32 = 0

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

///*
/// Specifies the Squared Euclidean Distance function.
struct CoreML_Specification_SquaredEuclideanDistance {
  // SwiftProtobuf.Message conformance is added in an extension below. See the
  // `Message` and `Message+*Additions` files in the SwiftProtobuf library for
  // methods supported on all messages.

  var unknownFields = SwiftProtobuf.UnknownStorage()

  init() {}
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "CoreML.Specification"

extension CoreML_Specification_KNearestNeighborsClassifier: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".KNearestNeighborsClassifier"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "nearestNeighborsIndex"),
    3: .same(proto: "numberOfNeighbors"),
    100: .same(proto: "stringClassLabels"),
    101: .same(proto: "int64ClassLabels"),
    110: .same(proto: "defaultStringLabel"),
    111: .same(proto: "defaultInt64Label"),
    200: .same(proto: "uniformWeighting"),
    210: .same(proto: "inverseDistanceWeighting"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularMessageField(value: &self._nearestNeighborsIndex) }()
      case 3: try { try decoder.decodeSingularMessageField(value: &self._numberOfNeighbors) }()
      case 100: try {
        var v: CoreML_Specification_StringVector?
        var hadOneofValue = false
        if let current = self.classLabels {
          hadOneofValue = true
          if case .stringClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.classLabels = .stringClassLabels(v)
        }
      }()
      case 101: try {
        var v: CoreML_Specification_Int64Vector?
        var hadOneofValue = false
        if let current = self.classLabels {
          hadOneofValue = true
          if case .int64ClassLabels(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.classLabels = .int64ClassLabels(v)
        }
      }()
      case 110: try {
        var v: String?
        try decoder.decodeSingularStringField(value: &v)
        if let v = v {
          if self.defaultClassLabel != nil {try decoder.handleConflictingOneOf()}
          self.defaultClassLabel = .defaultStringLabel(v)
        }
      }()
      case 111: try {
        var v: Int64?
        try decoder.decodeSingularInt64Field(value: &v)
        if let v = v {
          if self.defaultClassLabel != nil {try decoder.handleConflictingOneOf()}
          self.defaultClassLabel = .defaultInt64Label(v)
        }
      }()
      case 200: try {
        var v: CoreML_Specification_UniformWeighting?
        var hadOneofValue = false
        if let current = self.weightingScheme {
          hadOneofValue = true
          if case .uniformWeighting(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.weightingScheme = .uniformWeighting(v)
        }
      }()
      case 210: try {
        var v: CoreML_Specification_InverseDistanceWeighting?
        var hadOneofValue = false
        if let current = self.weightingScheme {
          hadOneofValue = true
          if case .inverseDistanceWeighting(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.weightingScheme = .inverseDistanceWeighting(v)
        }
      }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    try { if let v = self._nearestNeighborsIndex {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
    } }()
    try { if let v = self._numberOfNeighbors {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
    } }()
    switch self.classLabels {
    case .stringClassLabels?: try {
      guard case .stringClassLabels(let v)? = self.classLabels else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 100)
    }()
    case .int64ClassLabels?: try {
      guard case .int64ClassLabels(let v)? = self.classLabels else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 101)
    }()
    case nil: break
    }
    switch self.defaultClassLabel {
    case .defaultStringLabel?: try {
      guard case .defaultStringLabel(let v)? = self.defaultClassLabel else { preconditionFailure() }
      try visitor.visitSingularStringField(value: v, fieldNumber: 110)
    }()
    case .defaultInt64Label?: try {
      guard case .defaultInt64Label(let v)? = self.defaultClassLabel else { preconditionFailure() }
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 111)
    }()
    case nil: break
    }
    switch self.weightingScheme {
    case .uniformWeighting?: try {
      guard case .uniformWeighting(let v)? = self.weightingScheme else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    }()
    case .inverseDistanceWeighting?: try {
      guard case .inverseDistanceWeighting(let v)? = self.weightingScheme else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 210)
    }()
    case nil: break
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_KNearestNeighborsClassifier, rhs: CoreML_Specification_KNearestNeighborsClassifier) -> Bool {
    if lhs._nearestNeighborsIndex != rhs._nearestNeighborsIndex {return false}
    if lhs._numberOfNeighbors != rhs._numberOfNeighbors {return false}
    if lhs.classLabels != rhs.classLabels {return false}
    if lhs.defaultClassLabel != rhs.defaultClassLabel {return false}
    if lhs.weightingScheme != rhs.weightingScheme {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_NearestNeighborsIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".NearestNeighborsIndex"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "numberOfDimensions"),
    2: .same(proto: "floatSamples"),
    100: .same(proto: "linearIndex"),
    110: .same(proto: "singleKdTreeIndex"),
    200: .same(proto: "squaredEuclideanDistance"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.numberOfDimensions) }()
      case 2: try { try decoder.decodeRepeatedMessageField(value: &self.floatSamples) }()
      case 100: try {
        var v: CoreML_Specification_LinearIndex?
        var hadOneofValue = false
        if let current = self.indexType {
          hadOneofValue = true
          if case .linearIndex(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.indexType = .linearIndex(v)
        }
      }()
      case 110: try {
        var v: CoreML_Specification_SingleKdTreeIndex?
        var hadOneofValue = false
        if let current = self.indexType {
          hadOneofValue = true
          if case .singleKdTreeIndex(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.indexType = .singleKdTreeIndex(v)
        }
      }()
      case 200: try {
        var v: CoreML_Specification_SquaredEuclideanDistance?
        var hadOneofValue = false
        if let current = self.distanceFunction {
          hadOneofValue = true
          if case .squaredEuclideanDistance(let m) = current {v = m}
        }
        try decoder.decodeSingularMessageField(value: &v)
        if let v = v {
          if hadOneofValue {try decoder.handleConflictingOneOf()}
          self.distanceFunction = .squaredEuclideanDistance(v)
        }
      }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    // The use of inline closures is to circumvent an issue where the compiler
    // allocates stack space for every if/case branch local when no optimizations
    // are enabled. https://github.com/apple/swift-protobuf/issues/1034 and
    // https://github.com/apple/swift-protobuf/issues/1182
    if self.numberOfDimensions != 0 {
      try visitor.visitSingularInt32Field(value: self.numberOfDimensions, fieldNumber: 1)
    }
    if !self.floatSamples.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.floatSamples, fieldNumber: 2)
    }
    switch self.indexType {
    case .linearIndex?: try {
      guard case .linearIndex(let v)? = self.indexType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 100)
    }()
    case .singleKdTreeIndex?: try {
      guard case .singleKdTreeIndex(let v)? = self.indexType else { preconditionFailure() }
      try visitor.visitSingularMessageField(value: v, fieldNumber: 110)
    }()
    case nil: break
    }
    try { if case .squaredEuclideanDistance(let v)? = self.distanceFunction {
      try visitor.visitSingularMessageField(value: v, fieldNumber: 200)
    } }()
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_NearestNeighborsIndex, rhs: CoreML_Specification_NearestNeighborsIndex) -> Bool {
    if lhs.numberOfDimensions != rhs.numberOfDimensions {return false}
    if lhs.floatSamples != rhs.floatSamples {return false}
    if lhs.indexType != rhs.indexType {return false}
    if lhs.distanceFunction != rhs.distanceFunction {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_UniformWeighting: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".UniformWeighting"
  static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let _ = try decoder.nextFieldNumber() {
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_UniformWeighting, rhs: CoreML_Specification_UniformWeighting) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_InverseDistanceWeighting: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".InverseDistanceWeighting"
  static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let _ = try decoder.nextFieldNumber() {
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_InverseDistanceWeighting, rhs: CoreML_Specification_InverseDistanceWeighting) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_LinearIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".LinearIndex"
  static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let _ = try decoder.nextFieldNumber() {
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_LinearIndex, rhs: CoreML_Specification_LinearIndex) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_SingleKdTreeIndex: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".SingleKdTreeIndex"
  static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "leafSize"),
  ]

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      // The use of inline closures is to circumvent an issue where the compiler
      // allocates stack space for every case branch when no optimizations are
      // enabled. https://github.com/apple/swift-protobuf/issues/1034
      switch fieldNumber {
      case 1: try { try decoder.decodeSingularInt32Field(value: &self.leafSize) }()
      default: break
      }
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if self.leafSize != 0 {
      try visitor.visitSingularInt32Field(value: self.leafSize, fieldNumber: 1)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_SingleKdTreeIndex, rhs: CoreML_Specification_SingleKdTreeIndex) -> Bool {
    if lhs.leafSize != rhs.leafSize {return false}
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}

extension CoreML_Specification_SquaredEuclideanDistance: SwiftProtobuf.Message, SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  static let protoMessageName: String = _protobuf_package + ".SquaredEuclideanDistance"
  static let _protobuf_nameMap = SwiftProtobuf._NameMap()

  mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let _ = try decoder.nextFieldNumber() {
    }
  }

  func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try unknownFields.traverse(visitor: &visitor)
  }

  static func ==(lhs: CoreML_Specification_SquaredEuclideanDistance, rhs: CoreML_Specification_SquaredEuclideanDistance) -> Bool {
    if lhs.unknownFields != rhs.unknownFields {return false}
    return true
  }
}
