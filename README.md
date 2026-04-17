# superred-modules

Optimizer, target, and security claim implementations for the [superred](../superred/) framework.

## Structure

Each module is its own installable Python package inside one of three category folders:

```
optimizers/
  test_basic_prompt_list/     # pip: basic-prompt-list-optimizer
  test_hint_following/        # pip: hint-following-optimizer
  test_llm_prompt_generator/  # pip: llm-prompt-generator-optimizer
targets/
  test_basic_llm_chat/        # pip: basic-llm-chat-target
  test_filter_test/           # pip: filter-test-target
security_claims/
  test_basic_secret_leak/     # pip: basic-secret-leak-claim
  test_all_keys_match/        # pip: all-keys-match-claim
```

## Setup

Requires Python 3.11-3.13 and the shared venv at the repo root.

```bash
# Activate the shared venv
source ../.venv/bin/activate

# Install the framework first (editable)
pip install -e ../superred

# Install individual modules (editable)
pip install -e optimizers/test_basic_prompt_list
pip install -e targets/test_basic_llm_chat
pip install -e security_claims/test_basic_secret_leak
```

Some modules have additional dependencies (e.g. `basic-llm-chat-target` requires `litellm`). These are declared in each module's `pyproject.toml` and installed automatically by pip.

## Usage

After installation, import by package name:

```python
from basic_prompt_list_optimizer import BasicPromptListOptimizer
from basic_llm_chat_target import BasicLLMChatTarget, USER_INPUT_TAG
from basic_secret_leak_claim import basic_secret_leak_claim
```

## Adding a new module

1. Create a directory under the appropriate category with `test_` prefix (e.g. `optimizers/test_my_optimizer/`)
2. Add `pyproject.toml` with `superred` as a dependency
3. Add `src/<package_name>/` with `__init__.py` and implementation
4. Install with `pip install -e optimizers/test_my_optimizer`
