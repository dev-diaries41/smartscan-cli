import torch
import open_clip
from onnx_utils.textencoder import TextEncoder
from onnxruntime.quantization import quantize_dynamic, QuantType
import os


def image_encoder_to_onnx(model_name: str, pretrained: str, output_path: str):
    """
    Converts a CLIP Image encoder model to ONNX format.

    Args:
        model_name (str): Name of the CLIP model (e.g., 'ViT-B-32').
        pretrained (str): Pretrained weights to use.
        output_path (str): Path to save the ONNX model.
    """
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    
    # Create a dummy input tensor of shape [1, 3, 224, 224]
    sample_input = torch.randn(1, 3, 224, 224, device="cpu")
    tmp_model_path = f"{os.path.splitext(output_path)[0]}_temp.onnx"

    torch.onnx.export(
        model.visual.eval(),
        sample_input,
        tmp_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    quantize_dynamic(tmp_model_path, output_path, weight_type=QuantType.QInt8,
                  nodes_to_exclude=['/conv1/Conv'])

    print(f"Quantized model successfully exported to {output_path}")

    os.remove(tmp_model_path) # only save quantized model


def text_encoder_to_onnx(model_name: str, pretrained: str, output_path: str):
    """
    Converts a CLIP TextEncoder model to ONNX format.

    Args:
        model_name (str): Name of the CLIP model (e.g., 'ViT-B-32').
        pretrained (str): Pretrained weights to use.
        output_path (str): Path to save the ONNX model.
    """
    # Load the CLIP model
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    
    # Initialize the text encoder with appropriate parameters
    text_encoder = TextEncoder(
        embed_dim=512,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12
    )
    
    # Load state dictionary from the CLIP model
    text_encoder.load_state_dict(model.state_dict(), strict=False)
    text_encoder.eval()
    
    # Get tokenizer and sample input tensor
    tokenizer = open_clip.get_tokenizer(model_name)
    input_tensor = tokenizer(["a diagram"]).to(device="cpu")

    tmp_model_path = f"{os.path.splitext(output_path)[0]}_temp.onnx"

    
    # Convert model to ONNX
    torch.onnx.export(
        text_encoder, input_tensor, tmp_model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    quantize_dynamic(tmp_model_path, output_path, weight_type=QuantType.QInt8)
    print(f" Quantized model successfully exported to {output_path}")

    os.remove(tmp_model_path) # only save quantized model

# if __name__ == "__main__":
#     # image_encoder_to_onnx(MODEL_NAME, PRETRAINED, IMAGE_ENCODER_PATH)
#     text_encoder_to_onnx(MODEL_NAME, PRETRAINED, TEXT_ENCODER_PATH)
