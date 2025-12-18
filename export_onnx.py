import torch
from unet.net import UNet
import os


def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(3, 8).to(device)

    model_path = "unet_model.pth"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"Warning: {model_path} not found. Exporting initialized model (random weights)."
        )

    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    output_path = "unet_model.onnx"
    print(f"Exporting to {output_path}...")

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Export complete! Model saved to {output_path}")


if __name__ == "__main__":
    export_onnx()
