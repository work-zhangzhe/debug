import onnx
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as rt
import os


def run():
    # load onnx model
    onnx_model = onnx.load("resnet18.onnx")

    if onnx_model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return

    if not os.path.exists("weight"):
        os.mkdir("weight")
    if not os.path.exists("bias"):
        os.mkdir("bias")
    if not os.path.exists("per_layer_output"):
        os.mkdir("per_layer_output")

    # remove initializer from input
    inputs = onnx_model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnx_model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    cnt = 0
    for initializer in onnx_model.graph.initializer:
        if len(initializer.dims) == 4:
            if len(initializer.raw_data) == 0:
                weight = np.array(initializer.float_data, dtype=np.float32).reshape(-1)
            else:
                weight = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(
                    -1
                )
            idx = [1 for _ in range(len(weight))]

            plt.scatter(weight, idx, s=0.1)
            plt.savefig(f"weight/{cnt}_" + initializer.name.replace("/", "_") + ".png")
            plt.cla()
            plt.clf()
            cnt += 1

    cnt = 0
    for initializer in onnx_model.graph.initializer:
        if len(initializer.dims) == 1:
            if len(initializer.raw_data) == 0:
                bias = np.array(initializer.float_data, dtype=np.float32).reshape(-1)
            else:
                bias = np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(-1)
            idx = [1 for _ in range(len(bias))]

            plt.scatter(bias, idx, s=0.1)
            plt.savefig(f"bias/{cnt}_" + initializer.name.replace("/", "_") + ".png")
            plt.cla()
            plt.clf()
            cnt += 1

    # add output node for per layer
    for node in onnx_model.graph.node:
        for output in node.output:
            onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    sess.set_providers(["CPUExecutionProvider"])

    # get input and output
    output_names = [output.name for output in sess.get_outputs()]

    input_tensor = {
        "img": np.fromfile("./noise.bin", dtype=np.float32).reshape(1, 3, 64, 64),
        "cond": np.fromfile("./cond.bin", dtype=np.float32).reshape(1, 3, 64, 64),
        "temb": np.fromfile("./temb.bin", dtype=np.float32).reshape(1, 128, 1, 1),
    }

    # session run
    outputs = sess.run(
        output_names,
        input_tensor,
    )

    for i, (output_name, output) in enumerate(zip(output_names, outputs)):
        try:
            output = output.reshape(-1)
            idx = [1 for _ in range(len(output))]

            plt.scatter(output, idx, s=0.1)
            plt.savefig(
                f"per_layer_output/{i}_" + output_name.replace("/", "_") + ".png"
            )
            plt.cla()
            plt.clf()
        except:
            continue


if __name__ == "__main__":
    run()
