# goal_passthrough -- assumptions and deviations

The direct-request baseline injects `Goal.description` verbatim into the first
user-facing controllable, once, then ends.

## Free-text gating (DTAP fitness)

`_handle_pre_call` now injects only a controllable whose `value_type` is unstructured
text (text/str/string/html/markdown). On a target where the first non-`system_prompt`
controllable is schema-typed (e.g. DTAP `skill` / `tool_description_*`, json), writing
the raw goal string would be a silent no-op -- the target parses it, gets nothing, and
records a scored-but-empty baseline run. Declining such a surface lets the baseline land
on the genuine user-message channel instead. It remains the zero-attack reference point:
it never edits a tool return or a system prompt, only fills the user turn. A DTAP scope
with no user-message channel has nothing for the baseline to fill and should be excluded
rather than charted as a baseline.
