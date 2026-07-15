# Releasing modules to PyPI

This repo is a **monorepo of independent packages**. Each module directory has
its own `pyproject.toml` and is published to PyPI as its own package, on its own
schedule. Publishing one module never touches any other.

Authentication uses **PyPI Trusted Publishing** (OIDC): there are no API tokens
or secrets stored anywhere. GitHub proves to PyPI that an upload came from this
repository and this workflow.

## Naming

- **Dist name** (what people `pip install`): `superred-<kind>-<name>`, where
  `<kind>` is `target`, `optimizer`, or `claim`.
  Example: `superred-target-minimal-llm-chat`.
- **Import name** stays whatever the package's `src/` directory is called (for
  example `minimal_llm_chat`). Distribution name and import name differing is
  normal in Python (`pip install pillow`, `import PIL`).
- **Release tag**: `<dist-name>-v<version>`, e.g.
  `superred-target-minimal-llm-chat-v0.1.0`.

## One-time setup, per repository

Create two GitHub environments (Settings -> Environments -> New environment):

- `pypi`
- `testpypi` (only if you want TestPyPI rehearsals)

Nothing else. No secrets.

## One-time setup, per package

Do this once for each package the first time you publish it. Because the package
does not exist on PyPI yet, you register a **"pending" publisher**; PyPI creates
the project on the first successful upload. There is no "create project" button
on PyPI, and that is expected.

1. Log in to <https://pypi.org/>.
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Under **"Add a new pending publisher"**, select the **GitHub** tab.
4. Fill in exactly:

   | Field             | Value                          |
   | ----------------- | ------------------------------ |
   | PyPI Project Name | the dist name, e.g. `superred-target-minimal-llm-chat` |
   | Owner             | `RoldSI`                       |
   | Repository name   | `superred-modules`             |
   | Workflow name     | `release.yml`                  |
   | Environment name  | `pypi`                         |

5. Click **Add**.

Only the "PyPI Project Name" changes from package to package. Owner, repo,
workflow, and environment are the same for every module.

(Optional, for TestPyPI rehearsals: repeat at
<https://test.pypi.org/manage/account/publishing/> with environment `testpypi`.)

## Releasing a package

1. Bump `version` in that package's `pyproject.toml`, and merge it to `main`.
2. Tag the release and push the tag:

   ```bash
   git tag superred-target-minimal-llm-chat-v0.1.0
   git push origin superred-target-minimal-llm-chat-v0.1.0
   ```

That is it. The `Release` workflow then:

1. Reads the tag, finds the package directory whose `pyproject` name matches.
2. Fails loudly if the tag version and the `pyproject` version disagree.
3. Builds **only** that package, runs `twine check`.
4. Publishes it to PyPI via Trusted Publishing.

Watch it under the **Actions** tab. When green, the package is live at
`https://pypi.org/project/<dist-name>/`.

## Rehearsing on TestPyPI

Go to **Actions -> Release -> Run workflow**, enter the dist name, and run it.
It builds and uploads to TestPyPI instead of PyPI. TestPyPI is a throwaway
sandbox and is periodically wiped.

## Adding a new package to the release pipeline

Nothing to change in the workflow: it discovers packages by scanning every
`pyproject.toml` in the repo. A new module is releasable as soon as it has a
`pyproject.toml` with a unique `name`, a `LICENSE`, and a registered pending
publisher on PyPI.

Before a module's first release, make sure it has:

- a `superred-<kind>-<name>` dist name,
- a `LICENSE` file (MIT, matching the `superred` framework) plus `license = "MIT"`
  and `license-files = ["LICENSE"]` in `pyproject.toml`,
- `authors = [{ name = "Simon Sure", email = "info@simonsure.com" }]`,
- a `README.md` referenced by `readme = "README.md"` (it becomes the PyPI page),
- attribution for any vendored upstream code or bundled datasets (see
  `security_claims/strongreject/LICENSES/` for the pattern to copy).

## Troubleshooting

- **"No package named X in this repo"**: the tag's dist name does not match any
  `pyproject` `[project].name`. Check for a typo in the tag.
- **"Tag says version X, but pyproject says Y"**: bump the version in the
  package's `pyproject.toml` (or re-tag to match), then push the tag again.
- **"File already exists" from PyPI**: that version was already uploaded. PyPI
  versions are immutable; bump to a new version.
- **`invalid-publisher` / OIDC error**: the pending publisher on PyPI does not
  match. Recheck owner `RoldSI`, repo `superred-modules`, workflow `release.yml`,
  environment `pypi`, and that the PyPI Project Name equals the dist name.
