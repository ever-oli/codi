"""
ai.py — AI provider calls, prompt builders, and LaTeX-to-Unicode rendering.
"""
from __future__ import annotations

import difflib
import os


# ── Config (read at import time so callers don't need to pass env vars) ────────
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL",     "gemini-2.0-flash")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")
AI_PROVIDER      = os.environ.get("AI_PROVIDER",      "openrouter")


# ── LaTeX → Unicode ────────────────────────────────────────────────────────────
def render_latex(text: str) -> str:
    """
    Convert inline LaTeX math in AI responses to readable Unicode.
    Handles $...$, $$...$$, \\(...\\), and \\[...\\] delimiters.
    Falls back gracefully if pylatexenc is not installed.
    """
    try:
        import re
        from pylatexenc.latex2text import LatexNodes2Text
        conv = LatexNodes2Text(math_mode="text")

        def _replace(m: re.Match) -> str:
            inner = m.group(1) or m.group(2) or m.group(3)
            try:
                return conv.latex_to_text(inner)
            except Exception:
                return m.group(0)

        text = re.sub(r'\$\$(.+?)\$\$', _replace, text, flags=re.DOTALL)
        text = re.sub(r'\$(.+?)\$',     _replace, text, flags=re.DOTALL)
        text = re.sub(r'\\\((.+?)\\\)', _replace, text, flags=re.DOTALL)
        text = re.sub(r'\\\[(.+?)\\\]', _replace, text, flags=re.DOTALL)
    except ImportError:
        pass
    return text


# ── Providers ─────────────────────────────────────────────────────────────────
def _gemini(prompt: str) -> str:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No GEMINI_API_KEY found in .env."
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (result.text or "").strip()
    except Exception as e:
        return f"**Gemini error:** {e}"


def _openrouter(prompt: str) -> str:
    import json
    import urllib.request
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        return "No OPENROUTER_API_KEY found in .env."
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://github.com/ever-oli/codi",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"**OpenRouter error:** {e}"


def ai_call(prompt: str) -> str:
    if AI_PROVIDER == "gemini":
        return _gemini(prompt)
    return _openrouter(prompt)

# ── Helpers ───────────────────────────────────────────────────────────────
_EXT_TO_LANG = {
    ".py": "python", ".jl": "julia", ".R": "r",
}

def _lang_for(name: str) -> str:
    """Infer code-fence language from a problem filename."""
    import os
    _, ext = os.path.splitext(name)
    return _EXT_TO_LANG.get(ext, "python")

# ── Prompts ───────────────────────────────────────────────────────────────────
def get_hint(problem_name: str, ref_code: str, user_code: str | None) -> str:
    has_attempt = bool(user_code and user_code.strip())
    lang = _lang_for(problem_name)
    if has_attempt:
        diff_text = "\n".join(difflib.unified_diff(
            ref_code.splitlines(), (user_code or "").splitlines(),
            fromfile="reference", tofile="yours", lineterm="",
        ))
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

The student's current attempt diff vs reference:
```diff
{diff_text}
```

Give a single Socratic hint that nudges them toward what they're missing \
without revealing the answer. Be concise (2-4 sentences max). \
Focus on the most important gap in their attempt."""
    else:
        prompt = f"""\
You are a coding tutor helping a student study ML implementations from memory.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

The student hasn't written anything yet. Give a single Socratic hint to \
help them get started — what is the core concept or structure they need to \
think about? Be concise (2-4 sentences max). Do not give away the answer."""
    return ai_call(prompt)


def get_suggest_fix(problem_name: str, ref_code: str, user_code: str) -> str:
    lang = _lang_for(problem_name)
    diff_text = "\n".join(difflib.unified_diff(
        ref_code.splitlines(), user_code.splitlines(),
        fromfile="reference", tofile="yours", lineterm="",
    ))
    prompt = f"""\
You are a coding tutor reviewing a student's ML implementation attempt.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

Student's attempt diff vs reference:
```diff
{diff_text}
```

Be direct and specific. List exactly what is wrong and what needs to change \
to match the reference. Use short bullet points. Do not be vague. \
Do not explain concepts they already got right."""
    return ai_call(prompt)


def get_explain(problem_name: str, ref_code: str) -> str:
    lang = _lang_for(problem_name)
    prompt = f"""\
You are a coding tutor explaining an ML concept to a student who just finished \
(or attempted) an implementation exercise.

Problem: {problem_name}

Reference solution:
```{lang}
{ref_code}
```

Give a clear, concise explanation (5-8 sentences) of:
1. What this component does conceptually in the broader ML context
2. Why each key design decision in the implementation exists
3. One common real-world mistake or misconception to watch out for

Write for someone who can code but is still building intuition. \
Do not just restate the code line by line."""
    return ai_call(prompt)
