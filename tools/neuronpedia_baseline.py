"""Read-only local reproduction of Neuronpedia's oai_token-act-pair explainer.

Ported from hijohnnylin/neuronpedia commit
7724688596eb734a0662f911bf183151a5c66b2f. It is used only when an exact
existing GPT-5 baseline is absent; generated explanations are never uploaded.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from openai import OpenAI


SOURCE_COMMIT = "7724688596eb734a0662f911bf183151a5c66b2f"

SYSTEM_MESSAGE = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match."""

FIRST_USER = """

Neuron 1
Activations:
<start>
t\t0
urt\t0
ur\t0
ro\t0
 is\t0
 fab\t0
ulously\t0
 funny\t0
 and\t0
 over\t0
 the\t0
 top\t0
 as\t0
 a\t0
 '\t0
very\t0
 sneaky\t0
'\t1
 but\t0
ler\t0
 who\t0
 excel\t0
s\t0
 in\t0
 the\t0
 art\t0
 of\t0
 impossible\t0
 disappearing\t6
/\t0
re\t0
app\t0
earing\t10
 acts\t0
<end>
<start>
esc\t0
aping\t9
 the\t4
 studio\t0
 ,\t0
 pic\t0
col\t0
i\t0
 is\t0
 warm\t0
ly\t0
 affecting\t3
 and\t0
 so\t0
 is\t0
 this\t0
 ad\t0
roit\t0
ly\t0
 minimalist\t0
 movie\t0
 .\t0
<end>

Same activations, but with all zeros filtered out:
<start>
'\t1
 disappearing\t6
earing\t10
<end>
<start>
aping\t9
 the\t4
 affecting\t3
<end>

Explanation of neuron 1 behavior: the main thing this neuron does is find"""

SECOND_USER = """

Neuron 2
Activations:
<start>
as\t0
 sac\t0
char\t0
ine\t0
 movies\t0
 go\t0
 ,\t0
 this\t0
 is\t0
 likely\t0
 to\t0
 cause\t0
 massive\t0
 cardiac\t0
 arrest\t10
 if\t0
 taken\t0
 in\t0
 large\t0
 doses\t0
 .\t0
<end>
<start>
shot\t0
 perhaps\t0
 '\t0
art\t0
istically\t0
'\t0
 with\t0
 handheld\t0
 cameras\t0
 and\t0
 apparently\t0
 no\t0
 movie\t0
 lights\t0
 by\t0
 jo\t0
aquin\t0
 b\t0
aca\t0
-\t0
as\t0
ay\t0
 ,\t0
 the\t0
 low\t0
-\t0
budget\t0
 production\t0
 swings\t0
 annoy\t0
ingly\t0
 between\t0
 vert\t0
igo\t9
 and\t0
 opacity\t0
 .\t0
<end>

Same activations, but with all zeros filtered out:
<start>
 arrest\t10
<end>
<start>
igo\t9
<end>

Explanation of neuron 2 behavior: the main thing this neuron does is find"""

THIRD_USER = """

Neuron 3
Activations:
<start>
the\t0
 sense\t0
 of\t0
 together\t3
ness\t7
 in\t0
 our\t0
 town\t1
 is\t0
 strong\t0
 .\t0
<end>
<start>
a\t0
 buoy\t0
ant\t0
 romantic\t0
 comedy\t0
 about\t0
 friendship\t0
 ,\t0
 love\t0
 ,\t0
 and\t0
 the\t0
 truth\t0
 that\t0
 we\t2
're\t4
 all\t3
 in\t7
 this\t10
 together\t5
 .\t0
<end>

Explanation of neuron 3 behavior: the main thing this neuron does is find"""

DEMONSTRATIONS = [
    (FIRST_USER, "present tense verbs ending in 'ing'."),
    (SECOND_USER, "words related to physical medical conditions."),
    (THIRD_USER, "phrases related to community."),
]

ANOMALIES = {"âĢĶ": "—", "âĢĵ": "–", "âĢľ": "“", "âĢĿ": "”", "âĢĺ": "‘", "âĢĻ": "’", "âĢĭ": " ", "Ġ": " ", "Ċ": "↵", "<0x0A>": "↵", "\n": "↵", "ĉ": "\t", "▁": " "}


def clean_token(token: str) -> str:
    for old, new in ANOMALIES.items():
        token = token.replace(old, new)
    return token


def activation_block(exemplar: dict[str, Any]) -> str:
    tokens = exemplar.get("full_tokens") or exemplar.get("tokens") or []
    values = exemplar.get("full_values") or exemplar.get("values") or []
    rows = "\n".join(f"{clean_token(str(token))}\t{value}" for token, value in zip(tokens, values))
    return f"\n  <start>\n{rows}\n<end>"


def build_messages(exemplars: list[dict[str, Any]]) -> list[dict[str, str]]:
    user = "\n        \nNeuron 4\nActivations:"
    user += "".join(activation_block(item) for item in exemplars[:10])
    user += '\nOnly respond with the explanation itself, which should not be a full sentence, just the completion of "the main thing..." sentence. Do NOT include the whole phrase "Explanation of neuron 4 behavior: the main thing this neuron does is find...". Do not mention "this neuron...".'
    user += "\nExplanation of neuron 4 behavior: the main thing this neuron does is find"
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_MESSAGE}]
    for demo_user, demo_assistant in DEMONSTRATIONS:
        messages.append({"role": "user", "content": demo_user})
        messages.append({"role": "assistant", "content": demo_assistant})
    messages.append({"role": "user", "content": user})
    return messages


def generate_baseline(exemplars: list[dict[str, Any]], model: str) -> dict[str, Any]:
    messages = build_messages(exemplars)
    response = OpenAI().chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        reasoning_effort="low",
    )
    description = (response.choices[0].message.content or "").strip()
    if not description:
        raise RuntimeError("Local Neuronpedia-compatible baseline returned an empty explanation")
    prompt_sha = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    usage = response.usage
    return {
        "description": description,
        "model_snapshot": model,
        "prompt_sha256": prompt_sha,
        "neuronpedia_source_commit": SOURCE_COMMIT,
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        },
    }
