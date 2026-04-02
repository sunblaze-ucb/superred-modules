# superred-modules

Optimizer, target, and security claim implementations for the [superred](../super-red-teaming/) framework.

## Structure

Each module is its own installable Python package inside one of three category folders:

```
optimizers/
  trivial_prompt/        # pip: trivial-prompt-optimizer
targets/
  simple_chat/           # pip: simple-chat-target
security_claims/
  secret_leak/           # pip: secret-leak-claim
```

## Setup

Requires Python 3.11-3.13 and the shared venv at the repo root.

```bash
# Activate the shared venv
source ../.venv/bin/activate

# Install the framework first (editable)
pip install -e ../super-red-teaming

# Install individual modules (editable)
pip install -e optimizers/trivial_prompt
pip install -e targets/simple_chat
pip install -e security_claims/secret_leak
```

Some modules have additional dependencies (e.g. `simple-chat-target` requires `litellm`). These are declared in each module's `pyproject.toml` and installed automatically by pip.

## Usage

After installation, import by package name:

```python
from trivial_prompt_optimizer import TrivialPromptOptimizer
from simple_chat_target import SimpleChatTarget, USER_INPUT_TAG
from secret_leak_claim import secret_leak_claim
```

## Adding a new module

1. Create a directory under the appropriate category (e.g. `optimizers/my_optimizer/`)
2. Add `pyproject.toml` with `superred` as a dependency
3. Add `src/<package_name>/` with `__init__.py` and implementation
4. Install with `pip install -e optimizers/my_optimizer`
