"""
根据不同模型类型加载模型

"""


def model_loader(path: str, type: str):
    model = None

    if type == "MDM":
        import torch
        from mdm.model.v2 import MDMModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MDMModel.from_pretrained(path).to(device)
    elif type == "ONNX":
        import onnxruntime as ort

        ort_session = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        model = ort_session
    elif type == "RKNN":
        from rknnlite.api import RKNNLite as RKNN
        class RKNN_model_container():
            def __init__(self, model_path, target=None, device_id=None) -> None:
                rknn = RKNN()
                rknn.load_rknn(model_path)
                ret = rknn.init_runtime()
                if ret != 0:
                    print('Init runtime environment failed!')
                    exit(ret)
                self.rknn = rknn

            def run(self, inputs):
                if self.rknn is None:
                    print("ERROR: rknn has been released")
                    return []

                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    pass
                else:
                    inputs = [inputs]

                result = self.rknn.inference(inputs=inputs, data_format=['nhwc'])

                return result

            def release(self):
                self.rknn.release()
                self.rknn = None

        model = RKNN_model_container(path)

    else:
        print(f"Unsupported model type: {type}")
    return model
