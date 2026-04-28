from transformers import LlamaConfig, LlamaForCausalLM

from src.core.layer_duplicator import build_model_with_layers


def build_tiny_llama() -> LlamaForCausalLM:
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config)


def test_layer_duplication_shares_weights_and_runs_forward() -> None:
    model = build_tiny_llama()
    dup_model = build_model_with_layers(model, [0, 1, 2, 2, 3])

    copied_layer = dup_model._new_layers[2]
    duplicated_layer = dup_model._new_layers[3]

    assert copied_layer.self_attn is not duplicated_layer.self_attn
    assert (
        copied_layer.self_attn.q_proj.weight.data_ptr()
        == duplicated_layer.self_attn.q_proj.weight.data_ptr()
    )
    assert copied_layer.self_attn.layer_idx == 2
    assert duplicated_layer.self_attn.layer_idx == 3

    input_ids = __import__("torch").tensor([[1, 2, 3, 4]])
    outputs = dup_model(input_ids=input_ids)
    assert outputs.logits.shape == (1, 4, 128)
