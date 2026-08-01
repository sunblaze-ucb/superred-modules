# Optimizers (the attackers)

An **optimizer** is the automated attacker. Given a harmful goal, it decides
what text to feed into the target system's **injection points** (the framework
calls these "controllables", e.g. the user message or, when allowed, the system
prompt), reads back what the system produced, and uses that to decide its next
move. One **run** is one attack attempt; some optimizers make many runs, some
hold a whole back-and-forth conversation inside a single run.

Most of the modules here are faithful re-implementations of published jailbreak
techniques. Each ships an `ASSUMPTIONS.md` recording the source paper, the
reference code, and every deliberate deviation; read it for the technical
detail. This page is the plain-language tour.

## Two patterns shared by almost all of them

1. **They escalate with access.** By default an attacker only controls the user
   message. If the threat model also grants control of the **system prompt**,
   most of these attacks automatically use it (a much more powerful foothold).
   They also read any facts they are allowed to see, such as the target's model
   name, and feed that into their strategy.
2. **The framework has the final say on success.** Many of these attacks run
   their own internal scorer (an AI rating 1-10, or a check for refusal
   phrases) to steer themselves, but the official verdict always comes from the
   security claim's judge, not the attacker's own opinion.

## Quick reference

| Optimizer | Single or multi-turn | Uses an attacker LLM | In one line |
|-----------|----------------------|----------------------|-------------|
| AutoDAN-Turbo | single-turn, many attempts | yes (write + score + summarize) | Learns and reuses a growing playbook of jailbreak tactics |
| Bijection Learning | single-turn | no (cipher built in code) | Teaches a made-up cipher, then asks in code |
| CodeChameleon | single-turn | no (fixed encryption) | Disguises the request as a code puzzle to solve |
| Crescendo | multi-turn | yes (write + self-score) | Eases in with a slowly escalating conversation |
| DRA | single-turn | no (fixed template) | Hides the request letter-by-letter, asks model to rebuild it |
| EIA | single-run agent environment injection | no (fixed HTML) | Hides malicious web elements so agents type private data into them |
| FITD | multi-turn | yes (build ladder) | Small agreements first, then escalates step by step |
| FlipAttack | single-turn | no (text reversal) | Reverses the text, tells the model how to un-flip it |
| GEPA | single-turn, many rounds | yes (reflect + rewrite) | An AI reflects on feedback and rewrites the prompt |
| GOAT | multi-turn | yes (drives conversation) | An attacker AI converses, picking tactics each turn |
| GPTFuzzer | single-turn, many queries | yes (mutate templates) | Fuzz-tests known jailbreak templates by mutating them |
| Libertas | single-turn, many prompts | yes (rank metadata) | Replays Pliny's model-specific L1B3RT4S prompts byte-faithfully |
| Many-Shot | single-turn | optional | Floods a long fake conversation of compliant examples |
| PAIR | multi-turn refinement | yes (write + self-score) | An AI rewrites its prompt from the last response |
| PoisonedRAG | single-turn | optional | Corrupts retrieved context so RAG answers with a planted wrong answer |
| TAP | tree search | yes (write + judge) | Grows and prunes a tree of candidate prompts |

## Real attacks

### AutoDAN-Turbo (`autodan-turbo-optimizer`)

Learns a growing playbook of jailbreak tricks over many attempts and reuses the
ones that worked best. An attacker AI writes a fresh prompt each attempt; after
each one a scorer rates how much the target complied, and whenever an attempt
beats the previous best, a summarizer distills what made it work into a named
"strategy" saved in a library. Later attempts retrieve the best saved strategies
to guide the next prompt. Single-turn, default 10 attempts. Source: Liu et al.,
*AutoDAN-Turbo* (ICLR 2025).

### Bijection Learning (`bijection-optimizer`)

Invents a random secret cipher, teaches it to the target in the same message
(with an alphabet table and worked examples), then writes the harmful request in
that cipher so safety training does not recognize it. Each attempt uses a fresh
random code (keep trying until one slips through). No attacker LLM: the cipher
is built by plain code. Single-turn, default 6 attempts. Source: Huang, Li,
Tang, *Endless Jailbreaks with Bijection Learning* (ICLR 2025).

### CodeChameleon (`superred-optimizer-code-chameleon`)

Disguises the harmful request as a programming puzzle: it scrambles the request
(e.g. shuffling word order by a "binary tree" rule) and hands the target both the
scrambled text and a matching decryption routine, asking it to complete a
`ProblemSolver` program. Framed as benign code completion, the model
reconstructs and answers it. Fixed encryption template, no attacker LLM (it does
have an internal judge as a fallback). Single-turn. See `ASSUMPTIONS.md` for the
citation.

### Crescendo (`crescendo-optimizer`)

Eases the target into harmful territory through a gradually escalating
conversation: it opens with a benign, related question, then ratchets up turn by
turn, each follow-up leaning on the model's own previous answers so the drift
feels natural. If the target refuses, it backs off and tries another angle, and
it rotates through several pre-written escalation styles. Multi-turn; default up
to 15 attempts. Uses an attacker LLM plus an internal turn-by-turn evaluator.
Source: Russinovich et al., *Crescendo* (USENIX Security 2025).

### DRA, Disguise and Reconstruction Attack (`superred-optimizer-dra`)

Hides the harmful request one character at a time inside ordinary-looking cover
words (each real letter marked in parentheses), then instructs the target to
extract those letters, rebuild the hidden question, and answer it. Because the
model assembles the request itself rather than reading it plainly, refusal
training is less likely to fire. Fixed template, no attacker LLM (lightweight
refusal checks plus an optional judge fallback). Single-turn, default 20
attempts. See `ASSUMPTIONS.md` for the citation.

### EIA, Environmental Injection Attack (`superred-optimizer-eia-agent`)

Poisons the web environment that an agent observes: it injects hidden or subtly
visible form fields into webpage/read content with labels like "this is the
right place to input the email address", causing the agent to type private data
into the attacker-controlled field while continuing the benign task. Fixed HTML
templates, no attacker LLM. Agent-focused; default is the official
action-grounding aria-label form injection. See `ASSUMPTIONS.md` for the
citation.

### FITD, Foot-in-the-Door (`superred-optimizer-fitd`)

Uses the classic persuasion trick: get the model to agree to small, benign
things first, then walk it up a ladder of progressively more harmful requests,
asking it to revise earlier answers toward each next step. Prior cooperation
makes each small step feel consistent. If a step is refused, it softens and
backtracks. Multi-turn; default a 10-rung ladder, up to 5 attempts. Uses a
helper LLM to build the ladder and rewrites. See `ASSUMPTIONS.md` for the
citation.

### FlipAttack (`superred-optimizer-flip-attack`)

Scrambles the request by reversing it (whole sentence, word order, or letters
within each word), so the harmful text never appears readably, then tells the
model how to un-flip it before answering. Pure text manipulation, no attacker
LLM (the LLM is used only as an internal judge). Single-turn; rotates through
four flip modes. See `ASSUMPTIONS.md` for the citation.

### GEPA (`gepa-optimizer`)

Treats the attack prompt as something to evolve: it sends a prompt, observes the
response and score, then asks a "reflection" AI to read that feedback and propose
an improved prompt, keeping a small pool of candidates and repeatedly mutating
the best one. Notably, it will claim the **system prompt** as its attack channel
when allowed (higher leverage). Single-turn rounds; default 20. Uses a
reflection LLM. Source: Agrawal et al., *GEPA* (2025); originally a
prompt-optimization method repurposed as an attacker.

### GOAT, Generative Offensive Agent Tester (`goat-optimizer`)

An attacker AI holds a free-form conversation with the target, and each turn
explicitly reasons (observe, think, choose a strategy, reply) about which of
seven known jailbreak techniques to try next, switching tactics based on the
target's reactions. Only the chosen reply is sent; the reasoning stays internal.
Multi-turn (default 5 turns per attempt), default 10 attempts. Source: Pavlova
et al., *GOAT* (2024).

### GPTFuzzer (`superred-optimizer-gptfuzzer`)

Starts from a library of human-written jailbreak templates (each with a slot for
the harmful goal) and fuzz-tests them: it mutates templates with five operations
(combine two, expand, generate-similar, rephrase, shorten) and uses a tree
search to decide which templates to mutate next based on what has been paying
off. Uses an LLM only to mutate templates; scores responses with the project's
trained classifier. Single-turn; large default query budget. See `ASSUMPTIONS.md`
for the citation.

### Libertas (`superred-optimizer-libertas`)

Replays the model-specific jailbreak prompts from Pliny's L1B3RT4S corpus,
prioritizing the matching vendor family from the target's model identity before
universal and cross-provider prompts, and trying one prompt per run. The
upstream files are pinned by commit and preserved
byte-for-byte—including combining marks, variation selectors, confusables, and
invisible Unicode—and strict mode changes only a reviewed upstream goal-input
surface. One helper-LLM call ranks prompt metadata for the exact target
model within those priority tiers; raw jailbreak bodies are never sent to the
helper, and invalid output falls back to source order. There is no internal
judge. Because the upstream
corpus is AGPL-3.0, this independently installable module is also
AGPL-3.0-only. See its `README.md` and `ASSUMPTIONS.md` for the exact parity
boundary.

### Many-Shot Jailbreaking (`superred-optimizer-many-shot`)

Floods the target with one very long prompt containing many invented
user/assistant exchanges in which the assistant happily complies with harmful
requests, then appends the real harmful goal at the end. The volume of compliant
examples conditions the model to keep playing along. Examples come from a bundled
dataset (optionally it can ask an LLM to fabricate more). Single-turn; default
256 examples. See `ASSUMPTIONS.md` for the citation.

### PAIR, Prompt Automatic Iterative Refinement (`pair-optimizer`)

An attacker AI proposes a prompt, the target answers, a judge rates how
jailbroken the answer is (1-10), and the attacker refines its next prompt from
that feedback, running several refinement conversations in parallel "streams" at
once. Multi-turn refinement; default 3 streams times 3 iterations. Uses an
attacker LLM and an internal judge. See `ASSUMPTIONS.md` for the citation.

### PoisonedRAG (`superred-optimizer-poisonedrag`)

Poisons a RAG target's knowledge base or retrieved-context surface with a small
set of adversarial documents, then asks the target question so those documents
push the answer toward an attacker-chosen incorrect answer. When no corpus
surface is writable, it can fall back to the official RAG prompt wrapper in the
user message. Single-turn, default 5 poison documents. Uses an LLM only when it
needs to generate poison documents on the fly. See `ASSUMPTIONS.md` for the
citation.

### TAP, Tree of Attacks with Pruning (`superred-optimizer-tap`)

An attacker AI grows a branching tree of candidate prompts: it generates several
branches, prunes ones that drift off-topic, tests the survivors on the target
(a judge scores each 1-10), keeps the best, and expands them another level,
repeating until a prompt scores a perfect 10 or it hits the depth limit. Tree
search; default depth and width 10. Uses an attacker LLM plus pruning and scoring
judges. See `ASSUMPTIONS.md` for the citation.

## Test fixtures

These are not real attacks; they exist to exercise the framework's plumbing and
make good minimal examples to read.

### `demo_prompt_list` (`superred-optimizer-demo-prompt-list`)

The simplest possible attacker: it tries a fixed, pre-written list of prompts one
per run and stops when the list is exhausted. No LLM, no learning, no reaction to
responses. The clearest starting point for understanding the optimizer interface.

### `test_hint_following` (`hint-following-optimizer`)

A test attacker that reads instructions hidden in what it is allowed to observe
(text of the form "inject VALUE into NAME") and injects exactly those values. It
exists to verify the framework's scope filtering: it can only succeed for
injection points whose hints are visible within its granted access. No LLM.

### `test_llm_prompt_generator` (`llm-prompt-generator-optimizer`)

A minimal attacker that asks an LLM to write an attack prompt each round, feeding
back the previous rounds' scores so the LLM can try to improve. Its real purpose
is to exercise the LLM-access path and the cost-budget enforcement. Single-turn,
default 10 rounds.
