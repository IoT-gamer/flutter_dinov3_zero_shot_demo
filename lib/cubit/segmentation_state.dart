part of 'segmentation_cubit.dart';

enum SegmentationStatus {
  initial,
  loadingModel,
  modelReady,
  creatingPrototype,
  prototypeReady,
  error,
}

@immutable
class SegmentationState extends Equatable {
  const SegmentationState({
    this.status = SegmentationStatus.initial,
    this.objectPrototype,
    this.overlayImage,
    this.detectedBox,
    this.cutoutPngPath,
    this.isSavingCutout = false,
    this.isSegmenting = false,
    this.similarityThreshold = 0.8,
    this.selectedInputSize = 320,
    this.showLargestAreaOnly = true,
    this.showSlider = false,
    this.errorMessage,
  });

  final SegmentationStatus status;
  final List<double>? objectPrototype;
  final ui.Image? overlayImage;
  final List<double>? detectedBox;
  final String? cutoutPngPath;
  final bool isSavingCutout;
  final bool isSegmenting;
  final double similarityThreshold;
  final int selectedInputSize;
  final bool showLargestAreaOnly;
  final bool showSlider;
  final String? errorMessage;

  SegmentationState copyWith({
    SegmentationStatus? status,
    List<double>? objectPrototype,
    ui.Image? overlayImage,
    List<double>? detectedBox,
    String? cutoutPngPath,
    bool? isSavingCutout,
    bool? isSegmenting,
    double? similarityThreshold,
    int? selectedInputSize,
    bool? showLargestAreaOnly,
    bool? showSlider,
    String? errorMessage,
    bool clearPrototype = false,
    bool clearOverlay = false,
    bool clearBox = false,
    bool clearCutout = false,
  }) {
    return SegmentationState(
      status: status ?? this.status,
      objectPrototype: clearPrototype
          ? null
          : objectPrototype ?? this.objectPrototype,
      overlayImage: clearOverlay ? null : overlayImage ?? this.overlayImage,
      detectedBox: clearBox ? null : detectedBox ?? this.detectedBox,
      cutoutPngPath: clearCutout ? null : cutoutPngPath ?? this.cutoutPngPath,
      isSavingCutout: isSavingCutout ?? this.isSavingCutout,
      isSegmenting: isSegmenting ?? this.isSegmenting,
      similarityThreshold: similarityThreshold ?? this.similarityThreshold,
      selectedInputSize: selectedInputSize ?? this.selectedInputSize,
      showLargestAreaOnly: showLargestAreaOnly ?? this.showLargestAreaOnly,
      showSlider: showSlider ?? this.showSlider,
      errorMessage: errorMessage ?? this.errorMessage,
    );
  }

  @override
  List<Object?> get props => [
    status,
    objectPrototype,
    overlayImage,
    detectedBox,
    cutoutPngPath,
    isSavingCutout,
    isSegmenting,
    similarityThreshold,
    selectedInputSize,
    showLargestAreaOnly,
    showSlider,
    errorMessage,
  ];
}
