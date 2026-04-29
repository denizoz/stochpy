import textwrap
import pytest
from pathlib import Path

from stochpy.prompt_registry import PromptRegistry


@pytest.fixture
def tmp_prompts(tmp_path: Path) -> PromptRegistry:
    """Create a temporary prompts directory with sample YAML files."""
    shared = tmp_path / "shared"
    shared.mkdir()

    (shared / "time_filter.yaml").write_text(textwrap.dedent("""\
        time_filter:
          version: "1.0"
          user: |
            Always use DATESBETWEEN for time ranges.
    """))

    app = tmp_path / "app"
    app.mkdir()

    (app / "spam.yaml").write_text(textwrap.dedent("""\
        check:
          version: "2.0"
          system: |
            You are a spam classifier.
          user: |
            Is this email spam?
            From: {{ sender }}
            Body: {{ body }}
          examples:
            - input: "Buy now!"
              output: '{"is_spam": true}'
    """))

    (app / "report.yaml").write_text(textwrap.dedent("""\
        generate:
          version: "1.1"
          user: |
            Generate report for {{ quarter }}.
          partials:
            - shared.time_filter
    """))

    return PromptRegistry(root=str(tmp_path) + "/")


def test_load_basic(tmp_prompts):
    prompt = tmp_prompts.load("app.spam.check")
    assert prompt.version == "2.0"
    assert "spam" in prompt.user.lower()
    assert prompt.system is not None


def test_load_renders_variables(tmp_prompts):
    prompt   = tmp_prompts.load("app.spam.check")
    messages = prompt.render(sender="test@example.com", body="Hello")
    user_msg = messages[-1]   # last message is always the rendered user query
    assert "test@example.com" in user_msg["content"]
    assert "Hello"             in user_msg["content"]


def test_load_includes_system_message(tmp_prompts):
    prompt   = tmp_prompts.load("app.spam.check")
    messages = prompt.render(sender="x", body="y")
    roles    = [m["role"] for m in messages]
    assert "system" in roles


def test_load_includes_examples(tmp_prompts):
    prompt   = tmp_prompts.load("app.spam.check")
    messages = prompt.render(sender="x", body="y")
    roles    = [m["role"] for m in messages]
    # examples inject user/assistant pairs
    assert roles.count("assistant") >= 1


def test_partial_composition(tmp_prompts):
    prompt = tmp_prompts.load("app.report.generate")
    assert "DATESBETWEEN" in prompt.user


def test_missing_section_raises(tmp_prompts):
    with pytest.raises(KeyError):
        tmp_prompts.load("app.spam.nonexistent")


def test_missing_file_raises(tmp_prompts):
    with pytest.raises(FileNotFoundError):
        tmp_prompts.load("nonexistent.file.section")


def test_reload_clears_cache(tmp_prompts, tmp_path):
    tmp_prompts.load("app.spam.check")
    assert len(tmp_prompts._cache) > 0
    tmp_prompts.reload()
    assert len(tmp_prompts._cache) == 0
