from src.workers.model_utils import apply_chat_template_fallback


class TokenizerWithoutTemplate:
    bos_token = "<s>"


def test_apply_chat_template_fallback_without_native_template() -> None:
    prompt = apply_chat_template_fallback(
        TokenizerWithoutTemplate(),
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Say hello"},
        ],
    )
    assert prompt.startswith("<s>")
    assert "SYSTEM: Be concise." in prompt
    assert prompt.rstrip().endswith("ASSISTANT:")
