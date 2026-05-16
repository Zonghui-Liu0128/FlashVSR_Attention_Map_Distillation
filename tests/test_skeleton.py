import importlib

def test_can_import_all_subpackages():
    for sub in ["configs", "models", "attn", "losses", "data", "pipelines", "train"]:
        importlib.import_module(f"flashvsr_b1.{sub}")
