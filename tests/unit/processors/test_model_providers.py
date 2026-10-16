from app.processors.models_processor import ModelsProcessor

TRT = ("TensorrtExecutionProvider", {"trt_fp16_enable": True})
CUDA = ("CUDAExecutionProvider", {"device_id": 0})


def _processor(providers):
    mp = ModelsProcessor.__new__(ModelsProcessor)
    mp.providers = providers
    return mp


def test_trt_options_are_model_specific():
    opts = {"trt_engine_cache_prefix": "FaceParser"}
    providers = _processor([TRT, CUDA, "CPUExecutionProvider"])._model_providers(
        "FaceParser", opts
    )
    assert providers == [
        ("TensorrtExecutionProvider", opts),
        CUDA,
        "CPUExecutionProvider",
    ]


def test_excluded_model_skips_tensorrt():
    providers = _processor([TRT, CUDA])._model_providers("SelfieMulticlass", {})
    assert providers == [CUDA]


def test_excluded_model_with_only_tensorrt_falls_back_to_cpu():
    providers = _processor([TRT])._model_providers("SelfieMulticlass", {})
    assert providers == ["CPUExecutionProvider"]
