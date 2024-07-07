import warnings
from copy import deepcopy
from os import path as osp
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch
from onnxconverter_common.float16 import convert_float_to_float16

# from onnxsim import simplify
from neosr.archs import build_network
from neosr.utils.options import parse_options


def load_net():
    # build_network
    print(f"\n-------- Attempting to build network [{args.network}].")
    if args.network is None:
        msg = "Please select a network using the -net option"
        raise ValueError(msg)
    net_opt = {"type": args.network}

    if args.network == "omnisr":
        net_opt["upsampling"] = args.scale
        net_opt["window_size"] = args.window

    if args.window:
        net_opt["window_size"] = args.window

    net = build_network(net_opt)
    load_net = torch.load(args.input, map_location=torch.device("cuda"))
    # find parameter key
    print("-------- Finding parameter key...")
    try:
        if "params-ema" in load_net:
            param_key = "params-ema"
        elif "params" in load_net:
            param_key = "params"
        elif "params_ema" in load_net:
            param_key = "params_ema"
        load_net = load_net[param_key]
    except:  # noqa: S110
        pass

    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith("module."):
            load_net[k[7:]] = v
            load_net.pop(k)

    # load_network and send to device
    net.load_state_dict(load_net, strict=True)
    net = net.to(device="cuda", non_blocking=True)
    print(f"-------- Successfully loaded network [{args.network}].")
    torch.cuda.empty_cache()

    return net


def assert_verify(onnx_model, torch_model) -> None:
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)
    # onnxruntime output prediction
    # NOTE: "CUDAExecutionProvider" errors if some nvidia libs
    # are not found, defaulting to cpu
    ort_session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().cpu().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # torch outputs
    torch_outputs = torch_model(dummy_input)

    # final assert - default tolerance values - rtol=1e-03, atol=1e-05
    np.testing.assert_allclose(
        torch_outputs.detach().cpu().numpy(), ort_outs[0], rtol=0.01, atol=0.001
    )
    print("-------- Model successfully verified.")


def to_onnx() -> None:
    # error if network can't be converted
    net_error = ["craft", "ditn"]
    if args.network in net_error:
        msg = f"Network [{args.network}] cannot be converted to ONNX."
        raise RuntimeError(msg)

    # load network and send to device
    model = load_net()
    # set model to eval mode
    model.eval()

    # set static or dynamic
    if args.static is not None:
        dummy_input = torch.randn(1, *args.static, requires_grad=True)
    else:
        dummy_input = torch.randn(1, 3, 20, 20, requires_grad=True)

    # dict for dynamic axes
    if args.static is None:
        dyn_axes = {
            "dynamic_axes": {
                "input": {0: "batch_size", 2: "width", 3: "height"},
                "output": {0: "batch_size", 2: "width", 3: "height"},
            },
            "input_names": ["input"],
            "output_names": ["output"],
        }
    else:
        dyn_axes = None

    # add _fp32 suffix to output str
    filename, extension = osp.splitext(args.output)  # noqa: PTH122
    output_fp32 = filename + "_fp32" + extension
    # begin conversion
    print("-------- Starting ONNX conversion (this can take a while)...")

    with torch.device("cpu"):
        # TODO: switch to dynamo_export once it supports ATen PixelShuffle
        # then torch.testing.assert_close for verification

        torch.onnx.export(
            model,
            dummy_input,
            output_fp32,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=False,
            **(dyn_axes or {}),
        )

    print("-------- Conversion was successful. Verifying...")
    # verify onnx
    load_onnx = onnx.load(output_fp32)
    torch.cuda.empty_cache()
    onnx.checker.check_model(load_onnx)
    print(
        f"-------- Model successfully converted to ONNX format. Saved at: {output_fp32}."
    )
    # verify outputs
    if args.nocheck is False:
        assert_verify(output_fp32, model)

    if args.optimize:
        print("-------- Running ONNX optimization...")
        filename, extension = osp.splitext(args.output)  # noqa: PTH122
        output_optimized = filename + "_fp32_optimized" + extension
        session_opt = onnxruntime.SessionOptions()
        # ENABLE_ALL can cause compatibility issues, leaving EXTENDED as default
        session_opt.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        session_opt.optimized_model_filepath = output_optimized
        # save
        onnxruntime.InferenceSession(output_fp32, session_opt)
        # verify
        onnx.checker.check_model(onnx.load(output_optimized))
        print(f"-------- Model successfully optimized. Saved at: {output_optimized}")

    if args.fp16:
        print("-------- Converting to fp16...")
        output_fp16 = filename + "_fp16" + extension
        # convert to fp16
        if args.optimize:
            to_fp16 = convert_float_to_float16(onnx.load(output_optimized))
        else:
            to_fp16 = convert_float_to_float16(load_onnx)
        # save
        onnx.save(to_fp16, output_fp16)
        # verify
        onnx.checker.check_model(onnx.load(output_fp16))
        print(
            f"-------- Model successfully converted to half-precision. Saved at: {output_fp16}."
        )

    if args.fulloptimization:
        msg = "ONNXSimplify has been temporarily disabled."
        raise ValueError(msg)
        """
        # error if network can't run through onnxsim
        opt_error = ["omnisr"]
        if args.network in opt_error:
            msg = f"Network [{args.network}] doesnt support full optimization."
            raise RuntimeError(msg)

        print("-------- Running full optimization (this can take a while)...")
        output_fp32_fulloptimized = filename + "_fp32_fullyoptimized" + extension
        output_fp16_fulloptimized = filename + "_fp16_fullyoptimized" + extension

        # run onnxsim
        if args.optimize:
            simplified, check = simplify(onnx.load(output_optimized))
        elif args.fp16:
            simplified, check = simplify(onnx.load(output_fp16))
        else:
            simplified, check = simplify(load_onnx)
        assert check, "Couldn't validate ONNX model."

        # save and verify
        if args.fp16:
            onnx.save(simplified, output_fp16_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp16_fulloptimized))
        else:
            onnx.save(simplified, output_fp32_fulloptimized)
            onnx.checker.check_model(onnx.load(output_fp32_fulloptimized))

        print(
            f"-------- Model successfully optimized. Saved at: {output_fp32_fulloptimized}\n"
        )
        """


if __name__ == "__main__":
    torch.set_default_device("cuda")
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = Path(Path(__file__) / osp.pardir).resolve()
    __, args = parse_options(root_path)
    to_onnx()
