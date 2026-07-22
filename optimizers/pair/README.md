# superred-optimizer-pair

PAIR (Prompt Automatic Iterative Refinement) jailbreak optimizer for the
[superred](https://superred.simonsure.com) red-teaming framework.

PAIR is a query-efficient, black-box jailbreak method: an attacker LLM
iteratively refines an adversarial prompt against a target model, using a judge
to score each attempt and feeding the target's response back as the next
iteration's context.

## Install

```bash
pip install superred-optimizer-pair
```

## Credits / upstream

This optimizer is a faithful reimplementation of **PAIR** (Prompt Automatic
Iterative Refinement) for the superred framework.

- **PAIR** - Chao et al., "Jailbreaking Black Box Large Language Models in
  Twenty Queries" (arXiv:2310.08419),
  https://github.com/patrickrchao/JailbreakingLLMs. MIT License, Copyright (c)
  2023 PAIR Team. The attacker system prompts, the GPT-judge prompt and
  `Rating: [[n]]` format, the init/feedback message formats, the GCG refusal
  keyword dictionary, and the runtime defaults are ported verbatim from this
  repository and are redistributed here under its MIT license.
- **Persuasion prompt examples** - the logical-appeal and authority-endorsement
  worked examples derive from Zeng et al., "How Johnny Can Persuade LLMs to
  Jailbreak Them" (arXiv:2401.06373), reaching this module through the PAIR
  repository.
- **Refusal dictionary** - the `gcg` judge's refusal-string list originates
  from GCG (Zou et al., arXiv:2307.15043,
  https://github.com/llm-attacks/llm-attacks, MIT, Copyright (c) 2023 Andy
  Zou), via PAIR.

Our superred integration code is MIT-licensed (see LICENSE and NOTICE).
