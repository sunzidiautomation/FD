"""Check FLAIR's assumptions about the diffusers SD3 API. No GPU, no weights.

FLAIR hooks into diffusers internals -- the attention processor protocol,
the block forward contract, the encode_prompt return shape. A diffusers
upgrade can change those silently, and the failure would only surface
after a 5GB download and minutes of GPU time.

Run this first:

    python scripts/verify_api.py

Every check is introspection only: nothing is downloaded, nothing runs on
a GPU.
"""

import inspect
import sys

CHECKS: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, "OK" if ok else "FAIL"))
    mark = "  OK  " if ok else "  FAIL"
    print(f"{mark}  {name}" + (f"  -- {detail}" if detail and not ok else ""))


def main() -> int:
    try:
        from diffusers import StableDiffusion3Pipeline
        from diffusers.models.attention import JointTransformerBlock
        from diffusers.models.attention_processor import (
            Attention,
            JointAttnProcessor2_0,
        )
    except ImportError as exc:
        print(f"  FAIL  diffusers import -- {exc}")
        return 1

    import diffusers

    print(f"diffusers {diffusers.__version__}\n")

    # --- pipeline.encode_components depends on this ----------------------
    params = inspect.signature(StableDiffusion3Pipeline.encode_prompt).parameters
    for needed in (
        "prompt",
        "prompt_2",
        "prompt_3",
        "do_classifier_free_guidance",
        "max_sequence_length",
    ):
        check(f"encode_prompt accepts '{needed}'", needed in params)

    returns = [
        line.strip()
        for line in inspect.getsource(
            StableDiffusion3Pipeline.encode_prompt
        ).splitlines()
        if line.strip().startswith("return")
    ]
    check(
        "encode_prompt returns prompt_embeds first of 4",
        bool(returns) and returns[0].count(",") == 3 and "prompt_embeds" in returns[0],
        str(returns),
    )

    # --- pipeline.generate depends on this -------------------------------
    check(
        "__call__ accepts 'callback_on_step_end'",
        "callback_on_step_end"
        in inspect.signature(StableDiffusion3Pipeline.__call__).parameters,
    )

    # --- patching.install_flair depends on this --------------------------
    check("Attention.get_processor exists", hasattr(Attention, "get_processor"))
    check("Attention.set_processor exists", hasattr(Attention, "set_processor"))

    # --- processor.FlairJointProcessor delegates with this signature -----
    proc_params = list(
        inspect.signature(JointAttnProcessor2_0.__call__).parameters
    )
    check(
        "processor signature (attn, hidden_states, encoder_hidden_states, ...)",
        proc_params[:4] == ["self", "attn", "hidden_states", "encoder_hidden_states"],
        str(proc_params),
    )

    # --- patching.bypass_blocks depends on this --------------------------
    block_params = list(inspect.signature(JointTransformerBlock.forward).parameters)
    check(
        "block.forward(hidden_states, encoder_hidden_states, ...)",
        block_params[:3] == ["self", "hidden_states", "encoder_hidden_states"],
        str(block_params),
    )

    block_returns = [
        line.strip()
        for line in inspect.getsource(JointTransformerBlock.forward).splitlines()
        if line.strip().startswith("return")
    ]
    check(
        "block.forward returns (encoder_hidden_states, hidden_states)",
        any("return encoder_hidden_states, hidden_states" in r for r in block_returns),
        str(block_returns),
    )

    failed = [name for name, status in CHECKS if status == "FAIL"]
    print()
    if failed:
        print(f"{len(failed)} of {len(CHECKS)} checks FAILED:")
        for name in failed:
            print(f"  - {name}")
        print("\nFLAIR's hooks do not match this diffusers version.")
        print("Pin the version in requirements.txt, or update the code it names.")
        return 1

    print(f"All {len(CHECKS)} checks passed -- FLAIR's hooks match this diffusers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
