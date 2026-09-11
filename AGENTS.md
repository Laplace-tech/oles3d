# OLES3D Agent Contract

## Notebook runtime

- Project root: `/home/anna/projects/oles3d`
- Project interpreter: `/home/anna/projects/oles3d/.venv/bin/python`
- VS Code kernel display name: `.venv (3.12.3)`
- Kernelspec name: `python3`
- Do not create a global OLES3D kernelspec or use another project's environment.
- Do not ask the learner to select or diagnose the kernel when the agent can verify it directly.

Before creating or handing off any notebook, verify all of the following:

```bash
test -x .venv/bin/python
.venv/bin/python -m pip --version
test -x .venv/bin/pip
.venv/bin/pip --version
.venv/bin/python -m pip check
```

`python -m pip` is the authoritative package check, but the plain `.venv/bin/pip`
launcher must also exist because editor integrations may probe it. If the module
works and only the launcher is missing, verify the project-local `pip3` shebang
and restore only the missing launcher. If the module itself is missing, use
project-local `ensurepip`; do not recreate the environment or blindly reinstall
packages. When intentionally creating a new environment with `uv`, use a seeded
environment so that pip is present.

Also import `ipykernel`, `jupyter_client`, `torch`, and every package required by
the notebook. For deep-learning notebooks, verify CUDA availability and execute
a small CUDA tensor operation in a fresh process.

## Notebook creation

- Every code cell begins with `# Cell N — descriptive title`.
- Leave the second line empty; begin imports, comments, or implementation on the
  third line.
- The first cell is `# Cell 0 — Project Imports` followed only by imports.
- Set notebook metadata to:

```json
{
  "kernelspec": {
    "display_name": ".venv (3.12.3)",
    "language": "python",
    "name": "python3"
  }
}
```

- Execute the import cell with the project interpreter from the notebook's own
  directory and save its successful execution count.
- Until code is requested, a planned learner cell contains only its descriptive
  `# Cell N — ...` title.
- Every notebook must run from top to bottom without hidden state from another
  notebook.
- Tensor parameters and returns must include both Python type annotations and
  inline shape contracts, for example `# [B, C, D, H, W]`.
- Learner code must include Korean comments immediately above each meaningful
  block to explain what is selected, transformed, measured, or decided and why.
  Comments should trace Tensor/data flow like the learner's sampling example,
  not merely repeat the Python syntax.
- Write Korean docstrings and comments in concise nominal style such as `선택`,
  `계산`, and `반환`; avoid sentence endings such as `~한다` and `~합니다`.
- Preserve learner-authored code, comments, outputs, and unrelated Git changes.

## Runtime validation

- Connect to a real fresh project kernel and execute code; process existence is
  not evidence of kernel health.
- Run notebook validation sequentially, never with parallel `nbconvert` jobs.
- For short study notebooks, execute every non-empty code cell in order from the
  notebook directory.
- For long training notebooks, smoke-test imports, model construction, tensor
  shapes, and one forward/backward step; state what was not fully executed.
- Shut down only the temporary validation kernel. Never terminate healthy VS Code
  kernels or broad groups of Python processes.
- Do not use `wsl --shutdown`, broad `pkill`, VS Code force termination, or blind
  package reinstall/upgrade.

## Study and publishing

- Brief every study response with a compact ASCII roadmap showing completed
  Parts, the current Part/Lesson, and overall/current-lesson progress.
- Follow this fixed tutoring sequence for every prerequisite learning cell:
  "roadmap briefing -> concept explanation -> one learner cell in chat ->
  learner types and runs it -> validation after 다음/고카이체인지 -> next cell".
- Make concept explanations content-rich but compact. Explain purpose, relevant
  mathematics, Tensor Shape/Data Flow, and expected interpretation with a small
  ASCII visual or worked example when useful.
- Provide exactly one new learning cell in chat and stop. Never pre-write that
  learner cell into the notebook or execute it on the learner's behalf.
- When starting a new notebook, the agent may create only the notebook scaffold,
  set the project kernelspec, execute and save Cell 0 — Project Imports, and
  leave the next learner cell as its title only.
- Advance only after inspecting the learner's actual saved cell and output.
  Validate what was executed; do not assume that supplied code was entered.
- Write or execute learner code only when the learner explicitly asks for direct
  repair, automation, or implementation. A plain 다음 or 고카이체인지 is not
  such authorization.
- On 고카이체인지 <team>, briefly give the requested transformation call and
  matching emoji flourish, then follow the same validation and teaching
  sequence. The role-play must not replace technical explanation.
- On `다음`, inspect the actual saved notebook and show overall and current-lesson
  progress before providing the next learner cell.
- On `스톱오버`, validate all changed study notebooks sequentially from fresh
  project kernels without rewriting learner cells.
- On `커리원`, update only confirmed progress, stage only intended code and README
  files explicitly, commit, push `main`, and verify local/remote commit equality.
