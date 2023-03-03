import onnx
import numpy as np
import matplotlib.pyplot as plt


def run():
    # load onnx model
    onnx_model = onnx.load("./resnet18.onnx")

    if onnx_model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return

    # remove initializer from input
    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    for initializer in onnx_model.graph.initializer:
        if len(initializer.dims) == 4:
            weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(-1)
            idx = [1 for i in range(len(weight))]

            plt.scatter(weight, idx, s=0.1)
            plt.savefig(initializer.name + ".png")
            plt.cla()
            plt.clf()

        if len(initializer.dims) == 1:
            bias = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(-1)
            idx = [1 for i in range(len(bias))]

            plt.scatter(bias, idx, s=0.1)
            plt.savefig(initializer.name + ".png")
            plt.cla()
            plt.clf()


if __name__ == "__main__":
    run()
