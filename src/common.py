# common.py: shared code for SFT, GRPO, DPO, evaluation and the backend.
# The prompt, the response modes, the strict scorer and the rewards are defined ONCE here
# so training and evaluation can never silently disagree.
import math, os, re, json, hashlib, unicodedata, zlib
from collections import Counter, defaultdict
import numpy as np
from fractions import Fraction
from functools import lru_cache

try:
    from math_verify import LatexExtractionConfig, ExprExtractionConfig, parse, verify
except ImportError:  # allows importing for pure-python checks without math-verify
    parse = verify = LatexExtractionConfig = ExprExtractionConfig = None
try:     # math-verify's TimeoutException subclasses BaseException, so `except Exception` does NOT catch it
    from math_verify.errors import TimeoutException as _MVTimeout
except ImportError:
    class _MVTimeout(BaseException):
        pass
MV_ERRORS = (Exception, _MVTimeout)   # every math-verify call: a timeout = "not verified", never a crash
# math-verify enforces timeouts with signals on Linux (Colab); on Windows it would spawn processes.
import threading


def _to():
    """5 s math-verify timeout on Linux main thread; disabled where signals are unavailable."""
    if os.name == "nt" or threading.current_thread() is not threading.main_thread():
        return None
    return 5

# --------------------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------------------
# One system prompt for every trained v2 policy (SFT, GRPO, DPO) and for their evaluation.
POLICY_SYSTEM = (
    "You are a careful mathematical reasoner. Choose exactly one response mode. "
    "For problems that need deliberation, write <think> your step-by-step reasoning </think> "
    "and then the final answer in \\boxed{}. For problems you can answer directly, write "
    "<short> a brief solution ending with the final answer in \\boxed{} </short>."
)
# Official Qwen2.5-Math CoT prompt, used ONLY for the untouched base-model baseline (gate G0).
OFFICIAL_QWEN_MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
# DeepSeek-R1-Distill recommendation: no system prompt, instruction appended to the user turn.
R1_USER_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
# v1 adapters were trained on raw ChatML without a system prompt.
V1_TEMPLATE = "<|im_start|>user\n{q}\n<|im_end|>\n<|im_start|>assistant\n"

PROMPT_STYLES = ("policy", "official", "r1", "v1")


def build_prompt(tokenizer, question, style="policy", prefill=""):
    """Return the full prompt string (chat template applied). `prefill` forces a mode."""
    if style == "v1":
        return V1_TEMPLATE.format(q=question) + prefill
    if style == "policy":
        msgs = [{"role": "system", "content": POLICY_SYSTEM}, {"role": "user", "content": question}]
    elif style == "official":
        msgs = [{"role": "system", "content": OFFICIAL_QWEN_MATH_SYSTEM}, {"role": "user", "content": question}]
    elif style == "r1":
        msgs = [{"role": "user", "content": question + R1_USER_SUFFIX}]
    elif style == "thinkless":      # the Thinkless model card: instruction first, then the problem, one user turn
        msgs = [{"role": "user", "content": R1_USER_SUFFIX.strip() + "\n" + question}]
    else:
        raise ValueError(style)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefill


def text_hash(text):
    text = " ".join(unicodedata.normalize("NFC", str(text)).split())
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# --------------------------------------------------------------------------------------
# Response modes
# --------------------------------------------------------------------------------------
TAG_RE = re.compile(r"</?(?:think|short)>", re.I)
SHORT_RE = re.compile(r"\s*<short>(.*?)</short>\s*", re.S | re.I)
THINK_RE = re.compile(r"\s*<think>(.*?)</think>(.*)", re.S | re.I)


def parse_policy_output(text):
    """Classify a v2-policy completion.
    Returns (mode, answer_region, think_text):
      short     : <short> ... </short> and nothing else
      think     : <think> ... </think> followed by a non-empty answer region
      untagged  : no mode tags at all
      malformed : anything else (unclosed/truncated think, nested or extra tags, text after </short>)
    """
    t = str(text)
    m = SHORT_RE.fullmatch(t)
    if m and not TAG_RE.search(m[1]) and m[1].strip():
        return "short", m[1].strip(), ""
    m = THINK_RE.fullmatch(t)
    if m and not TAG_RE.search(m[1]) and not TAG_RE.search(m[2]) and m[1].strip() and m[2].strip():
        return "think", m[2].strip(), m[1].strip()
    if not TAG_RE.search(t):
        return "untagged", t.strip(), ""
    return "malformed", None, ""


def parse_plain_output(text):
    """Base / reference models: answer region is the text after the last </think> if present."""
    t = str(text)
    if re.search(r"</think>", t, re.I):
        head, tail = re.split(r"</think>", t, flags=re.I)[-2:]
        return "think", tail.strip(), head
    if re.search(r"<think>", t, re.I):  # opened but never closed = truncated reasoning
        return "malformed", None, t
    return "untagged", t.strip(), ""


def parse_hybrid_output(text):
    """Published hybrid models (prior-work baselines). Thinkless opens with <short> (unclosed, as in its
    Hybrid-OpenThoughts2 training data) or <think> ... </think>; AdaptThink's NoThinking mode closes an EMPTY
    thinking block (its chat template opens <think>). Mode = whether any reasoning precedes the last </think>."""
    t = str(text)
    low = t.lower()
    end = low.rfind("</think>")
    if end >= 0:
        thought = TAG_RE.sub("", t[:end]).strip()
        return ("think" if thought else "short"), t[end + len("</think>"):].strip(), thought
    if re.match(r"\s*<short>", t, re.I):
        return "short", TAG_RE.sub("", t).strip(), ""
    if "<think>" in low:                       # opened but never closed = truncated reasoning
        return "malformed", None, t
    return "untagged", t.strip(), ""


def classify(text, fmt):
    if fmt == "policy":
        return parse_policy_output(text)
    return parse_hybrid_output(text) if fmt == "hybrid" else parse_plain_output(text)


def format_ok(text):
    """Strict v2 format: exactly one mode, closed tags, final answer in \\boxed{}."""
    mode, region, _ = parse_policy_output(text)
    if mode not in ("think", "short"):
        return False
    boxes = boxed_answers(region)
    return bool(boxes) and bool(boxes[-1])

# --------------------------------------------------------------------------------------
# Strict scorer (from the validated v1 evaluation notebook, extended for modes and MC)
# --------------------------------------------------------------------------------------

def boxed_answers(text):
    """Balanced-brace \\boxed{...} extraction. Returns None for an unfinished box."""
    answers = []
    for m in re.finditer(r"\\boxed\s*\{", str(text)):
        start, depth, end = m.end(), 1, m.end()
        while end < len(text) and depth:
            ch = text[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            end += 1
        if depth:
            return None
        answers.append(text[start:end - 1].strip())
    return answers


def rational(text):
    s = str(text).strip().replace("\u2212", "-")
    s = re.sub(r"^\$|\$$", "", s).strip()
    s = re.sub(r"^\\\(|\\\)$", "", s).strip()
    s = re.sub(r"\\(?:text|mathrm)\{[^}]*\}", "", s).strip()   # units such as \text{ dollars}
    s = s.replace("\\$", "").replace("\\%", "").replace("%", "").strip()
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)            # thousands separators only
    frac = re.fullmatch(r"([+-]?)\\(?:d|t)?frac\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}", s)
    if frac:
        if int(frac[3]) == 0:
            return None
        return (-1 if frac[1] == "-" else 1) * Fraction(int(frac[2]), int(frac[3]))
    s = re.sub(r"\s+", "", s)
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+(?:\.\d*)?)?", s):
        return None
    try:
        parts = s.split("/")
        return Fraction(parts[0]) / Fraction(parts[1]) if len(parts) == 2 else Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _mv_parse(expr):
    return parse("\\boxed{" + expr + "}", extraction_config=[LatexExtractionConfig()],
                 fallback_mode="no_fallback", extraction_mode="first_match",
                 parsing_timeout=_to(), raise_on_error=True)


@lru_cache(maxsize=8192)
def symbolic_gold(gold):
    result = _mv_parse(gold)
    if not result:
        raise ValueError("Unparseable authoritative gold answer: " + repr(gold))
    return result


def extract_final(text, fmt="policy"):
    """Return (mode, answer_or_None, extraction_status)."""
    mode, region, _ = classify(text, fmt)
    if region is None:
        return mode, None, "malformed_or_truncated"
    boxes = boxed_answers(region)
    if boxes is None:
        return mode, None, "unfinished_box"
    if boxes:
        return (mode, boxes[-1], "boxed") if boxes[-1] else (mode, None, "empty_box")
    if mode == "short" and len(region) <= 40:   # e.g. <short>42</short>
        return mode, region, "short_text"
    return mode, None, "missing_final_answer"


NUMERIC_TASKS = ("gsm8k", "aime2026")

def score_text(text, gold, task, fmt="policy"):
    """Strict correctness of one completion. Failures stay in the denominator."""
    mode, answer, extraction = extract_final(text, fmt)
    res = {"correct": False, "final_answer": answer, "extraction": extraction, "mode": mode,
           "needs_review": False}
    if answer is None:
        res["status"] = "final_parse_failure"
        return res
    if task in NUMERIC_TASKS:
        expected = rational(gold)
        if expected is None:
            raise ValueError("Invalid numeric gold: " + repr(gold))
        actual = rational(answer)
        if actual is None and parse is not None:      # e.g. \boxed{18\text{ apples}} or 3\cdot 6
            try:
                res["correct"] = bool(verify(_mv_parse(str(gold)), _mv_parse(answer), strict=True,
                                             float_rounding=12, numeric_precision=30, timeout_seconds=_to()))
            except MV_ERRORS:
                res["correct"] = False
            res["status"] = "match" if res["correct"] else "final_parse_failure"
            res["needs_review"] = not res["correct"]
            return res
        res["correct"] = actual == expected
        res["status"] = "match" if res["correct"] else "numeric_mismatch"
        return res
    if task in ("math500", "math_train"):
        if rational(answer) is not None and rational(gold) is not None:
            res["correct"] = rational(answer) == rational(gold)
            res["status"] = "match" if res["correct"] else "numeric_mismatch"
            return res
        try:
            expected = symbolic_gold(str(gold))
            actual = _mv_parse(answer)
            if not actual:
                res.update(status="symbolic_parse_failure", needs_review=True)
                return res
            res["correct"] = bool(verify(expected, actual, strict=True, float_rounding=12,
                                         numeric_precision=30, timeout_seconds=_to(), raise_on_error=True))
            res.update(status="match" if res["correct"] else "symbolic_nonmatch",
                       needs_review=not res["correct"])
        except MV_ERRORS as exc:
            res.update(status="verifier_error", needs_review=True, verifier_error=repr(exc)[:200])
        return res
    raise ValueError("No validated scorer for task: " + task)


def second_scorer(text, gold, fmt="policy"):
    """Independent, lenient math-verify scorer (default extraction over the answer region).
    Used ONLY to report scorer agreement, never as the headline metric."""
    _, region, _ = classify(text, fmt)
    if region is None or parse is None:
        return False
    try:
        g = parse("$" + str(gold) + "$", parsing_timeout=_to())
        a = parse(region, parsing_timeout=_to())
        return bool(g and a and verify(g, a, timeout_seconds=_to()))
    except MV_ERRORS:
        return False

# --------------------------------------------------------------------------------------
# Repetition and trace statistics
# --------------------------------------------------------------------------------------

LOOP_MIN_REPS, LOOP_MIN_WORDS, LOOP_MAX_PERIOD = 5, 50, 200
LOOP_TAIL_CHARS, LOOP_TAIL_ZLIB = 3000, 0.15


def has_repetition_loop(text, min_reps=LOOP_MIN_REPS, min_words=LOOP_MIN_WORDS, max_period=LOOP_MAX_PERIOD,
                        tail_chars=LOOP_TAIL_CHARS, tail_zlib=LOOP_TAIL_ZLIB):
    """Degenerate-loop detector (v2, 2026-10-09). Flags (a) a block of 1..max_period words repeated back-to-back
    >= min_reps times and covering >= min_words words, anywhere in the text, or (b) a near-verbatim loop: the last
    tail_chars characters compress (zlib) below tail_zlib. Calibrated on the 12,205 full-scale S1 outputs: 0 of 8,928
    correct untruncated answers flagged, 52% of truncated ones. v1's rule (any word 4-gram seen 6 times, or one
    duplicated line) flagged 21% of normal base answers and 51% of R1-Distill answers: it measured length, not loops."""
    t = str(text)
    if len(t) >= tail_chars:
        tail = t[-tail_chars:].encode("utf-8", "ignore")
        if len(zlib.compress(tail, 9)) < tail_zlib * len(tail):
            return True
    words = t.split()
    n = len(words)
    if n < min_words:
        return False
    ids = {}
    a = np.fromiter((ids.setdefault(w, len(ids)) for w in words), dtype=np.int64, count=n)
    for p in range(1, max_period + 1):
        need = max((min_reps - 1) * p, min_words - p)     # run of equal words at lag p ⇔ (run + p) / p repetitions
        if need > n - p:
            break
        run = np.concatenate(([0], np.cumsum(a[p:] == a[:-p])))
        if (run[need:] - run[:-need] == need).any():
            return True
    return False


def chose_think(mode, text, template_opens_think=False):
    """The model's mode DECISION, used for every think-rate in the analysis: a closed think block, or a thinking block
    cut off before </think> (by the token limit). template_opens_think: the chat template itself opens <think>
    (R1-style models such as AdaptThink), so text without </think> is unfinished thinking. Format failures are
    reported separately (format_ok, truncated)."""
    if mode == "think":
        return True
    t = str(text).lstrip()
    if t.startswith("<think>"):
        return True
    return bool(template_opens_think) and "</think>" not in t


SELF_CHECK_RE = re.compile(r"\b(wait|let me (?:check|verify|re-?check)|double[- ]check|verify|hmm|"
                           r"actually|on second thought|let's check)\b", re.I)


def self_check_markers(text):
    return len(SELF_CHECK_RE.findall(str(text)))

# --------------------------------------------------------------------------------------
# GRPO reward functions (TRL signature: prompts, completions, **dataset columns)
# Dataset columns used: gold, task.  TRL passes completion_ids in recent versions.
# --------------------------------------------------------------------------------------
L_MAX = 2048            # = GRPO max_completion_length
LENGTH_ALPHA = 0.2


def _texts(completions):
    out = []
    for c in completions:
        if isinstance(c, list):               # conversational format fallback
            c = c[-1]["content"] if c else ""
        out.append(str(c))
    return out


def _ntoks(texts, kwargs):
    ids = kwargs.get("completion_ids")
    if ids is not None:
        return [len(x) for x in ids]
    return [max(1, int(len(t) / 3.6)) for t in texts]   # rough fallback, logged as such


def correctness_reward(prompts=None, completions=None, gold=None, task=None, **kwargs):
    """1 if the strictly scored final answer is correct, else 0 (truncated -> malformed -> 0)."""
    return [1.0 if score_text(t, g, k, "policy")["correct"] else 0.0
            for t, g, k in zip(_texts(completions), gold, task)]


def format_reward(completions=None, **kwargs):
    return [0.1 if format_ok(t) else -0.5 for t in _texts(completions)]


def repetition_reward(completions=None, **kwargs):
    return [-0.5 if has_repetition_loop(t) else 0.0 for t in _texts(completions)]


def mode_length_reward(prompts=None, completions=None, gold=None, task=None, **kwargs):
    """Standard length bonus (R3 ablation): CORRECT answers get more for fewer tokens, whatever the problem.
    Never rewards fast wrong answers."""
    texts = _texts(completions)
    lens = _ntoks(texts, kwargs)
    out = []
    for t, n, g, k in zip(texts, lens, gold, task):
        ok = score_text(t, g, k, "policy")["correct"]
        out.append(LENGTH_ALPHA * max(0.0, 1.0 - n / L_MAX) if ok else 0.0)
    return out


# C2 (ours, R1): DIFFICULTY-AWARE MODE REWARD. GRPO already samples a group of G answers per prompt; the group's
# pass rate p is a free, on-policy difficulty estimate (no difficulty model, no extra rollouts). Correct answers get
#   mode bonus   MODE_BETA · (p if <short> else 1 − p if <think>)  → short when the model finds it easy, think when hard
#   length bonus MODE_LEN_ALPHA · (n_max − n)/(n_max − n_min) among the group's correct answers in the SAME mode
# (0.5 · MODE_LEN_ALPHA if that mode has a single correct answer). Wrong answers get 0. The total bonus budget
# (0.1 + 0.1) equals the standard length bonus (LENGTH_ALPHA = 0.2), so R1 vs R3 isolates HOW the bonus is allocated.
MODE_BETA, MODE_LEN_ALPHA = 0.1, 0.1


def adaptive_bonus(groups, correct, modes, lens):
    """Difficulty-aware bonus per completion; `groups` = one key per completion (same key = same prompt)."""
    by = defaultdict(list)
    for i, g in enumerate(groups):
        by[g].append(i)
    out = [0.0] * len(correct)
    for idx in by.values():
        p = sum(bool(correct[i]) for i in idx) / len(idx)
        for mode in ("think", "short"):
            ok = [i for i in idx if correct[i] and modes[i] == mode]
            if not ok:
                continue
            lo, hi = min(lens[i] for i in ok), max(lens[i] for i in ok)
            for i in ok:
                rel = (hi - lens[i]) / (hi - lo) if hi > lo else 0.5
                out[i] = MODE_BETA * (p if mode == "short" else 1.0 - p) + MODE_LEN_ALPHA * rel
    return out


def group_keys(n, prompts=None, kwargs=None):
    """Group id per completion: the dataset `id` column (TRL passes it), else the prompt text, else one group."""
    ids = (kwargs or {}).get("id")
    if ids is not None and len(ids) == n:
        return list(ids)
    if prompts is not None and len(prompts) == n:
        return [str(p) for p in prompts]
    return [0] * n


def adaptive_mode_reward(prompts=None, completions=None, gold=None, task=None, **kwargs):
    texts = _texts(completions)
    correct = [score_text(t, g, k, "policy")["correct"] for t, g, k in zip(texts, gold, task)]
    modes = [parse_policy_output(t)[0] for t in texts]
    return adaptive_bonus(group_keys(len(texts), prompts, kwargs), correct, modes, _ntoks(texts, kwargs))


# Monitoring-only "rewards" (weight 0): TRL logs their means, giving think-rate etc. per step.
def think_rate(completions=None, **kwargs):
    """Mode decision, as in the evaluation (chose_think): thinking cut off at the token limit still counts."""
    return [1.0 if chose_think(parse_policy_output(t)[0], t) else 0.0 for t in _texts(completions)]


def think_correct(prompts=None, completions=None, gold=None, task=None, **kwargs):
    """1 if the rollout chose think AND is correct. Per step: think accuracy = think_correct / think_rate and
    short accuracy = (strict_accuracy − think_correct) / (1 − think_rate), without NaN rewards."""
    texts = _texts(completions)
    acc = correctness_reward(prompts, completions, gold, task, **kwargs)
    return [a if chose_think(parse_policy_output(t)[0], t) else 0.0 for t, a in zip(texts, acc)]


def policy_think_prob(completions=None, policy_think_prob=None, **kwargs):
    """Balanced runs (R1b) force the rollouts 50/50, so think_rate is fixed at 0.5. This logs the policy's OWN
    probability of choosing <think> for the prompt, measured by the rollout function (None if not measured; TRL
    ≥1.15 turns None into NaN, excluded from the logged mean, and the weight is 0)."""
    n = len(_texts(completions))
    return list(policy_think_prob) if policy_think_prob is not None else [None] * n


def all_correct_group(prompts=None, completions=None, gold=None, task=None, **kwargs):
    """1 for every rollout of a group whose rollouts are ALL strictly correct, else 0. Its per-step mean is the
    share of all-correct groups. In those groups only the shaping bonus varies, so group-std normalisation rescales
    the bonus to a unit-size advantage whatever its weight (the S4–S6 CoT collapse, C2f mechanism controls)."""
    keys = group_keys(len(_texts(completions)), prompts, kwargs)
    acc = correctness_reward(prompts, completions, gold, task, **kwargs)
    full = defaultdict(lambda: True)
    for k, a in zip(keys, acc):
        full[k] = full[k] and a == 1.0
    return [1.0 if full[k] else 0.0 for k in keys]


def short_rate(completions=None, **kwargs):
    return [1.0 if parse_policy_output(t)[0] == "short" else 0.0 for t in _texts(completions)]


def malformed_rate(completions=None, **kwargs):
    return [1.0 if parse_policy_output(t)[0] in ("malformed", "untagged") else 0.0
            for t in _texts(completions)]


def strict_accuracy(prompts=None, completions=None, gold=None, task=None, **kwargs):
    return correctness_reward(prompts, completions, gold, task, **kwargs)

# --------------------------------------------------------------------------------------
# LLM-judge reward (v1 judge, kept for the C3 study)
# --------------------------------------------------------------------------------------
JUDGE_PROMPT = """You are a strict, impartial grader. You will see a PROBLEM, a \
REFERENCE SOLUTION, and a CANDIDATE RESPONSE from another AI. Judge ONLY whether \
the candidate's final answer/approach is logically correct and complete, ignoring \
style. Output exactly one line: "SCORE: x.xx" where x.xx is a number from 0.00 \
(completely wrong) to 1.00 (fully correct and complete). No other text.

PROBLEM:
{problem}

REFERENCE SOLUTION:
{reference}

CANDIDATE RESPONSE:
{candidate}

SCORE:"""
JUDGE_DEFAULT_ON_ERROR = 0.5     # v1 behaviour
_JUDGE_NUM_RE = re.compile(r"(\d\.\d+|\d+\.\d+|\d+)")


def judge_prompt(problem, reference, candidate):
    return JUDGE_PROMPT.format(problem=str(problem)[:1500], reference=str(reference)[:1500],
                               candidate=str(candidate)[:1500])


def parse_judge_v1(text):
    """v1 parser (lenient): first number after </think>, clipped to [0,1]; None if absent."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = _JUDGE_NUM_RE.search(text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


_SCORE_LINE_RE = re.compile(r"SCORE\s*:\s*\**\s*(-?\d+(?:\.\d+)?)", re.I)
_LEADING_NUM_RE = re.compile(r"\s*\**\s*(-?\d+(?:\.\d+)?)(?![\d.])")


def _valid_score(x):
    v = float(x)
    return v if 0.0 <= v <= 1.0 else None      # an answer like 13535 or 70.00 is not a score


def parse_judge_fixed(raw):
    """Corrected parser: the score the judge actually gave (found in the pilot, 2026-10-08).
    The judge prompt ends with 'SCORE:', so the judge's verdict is the number that COMPLETES it, i.e. the first
    token of the output. The 1.5B R1 judge then writes '</think>' and re-solves the problem. v1 split on
    '</think>' and took the first number of that RE-SOLUTION, clipped to [0,1] → almost always 1.0 = "correct".
      1. a leading number completes 'SCORE:' → that is the verdict (invalid → None if outside [0,1])
      2. otherwise the last explicit 'SCORE: x' (after </think> when present)
    Returns None when the judge gave no valid score."""
    text = str(raw)
    lead = _LEADING_NUM_RE.match(text)
    if lead:
        return _valid_score(lead[1])
    region = re.split(r"</think>", text, flags=re.I)[-1] if re.search(r"</think>", text, re.I) else text
    found = _SCORE_LINE_RE.findall(region)
    return _valid_score(found[-1]) if found else None


def parse_judge_strict(raw):
    """Strict parser: only an exact 'SCORE: x' line (after an optional closed think block)."""
    cleaned = str(raw).strip()
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.I)[-1].strip()
    m = re.fullmatch(r"(?:SCORE:\s*)?(0(?:\.\d+)?|1(?:\.0+)?)", cleaned)
    return float(m[1]) if m else None

# --------------------------------------------------------------------------------------
# Multi-perspective reasoning with memory and validation (MPV): test-time method
# 1) K independent attempts, each from a different perspective   (diversity, no error propagation)
# 2) self-validation of every attempt -> VERDICT                   (verification)
# 3) selection: answer groups scored by votes + passed validations (no gold answer used)
# 4) synthesis with a memory of all attempts + verdicts            (final best solution)
# --------------------------------------------------------------------------------------
PERSPECTIVES = [
    ("direct", "Solve it directly, step by step."),
    ("alternative", "Solve it with a different method than the most obvious one (for example work backwards, "
                    "set up equations, or use a formula instead of enumeration)."),
    ("first-principles", "Solve it from first principles: restate what is given and what is asked, "
                         "then derive the answer carefully."),
    ("solve-and-verify", "Solve it, then verify the result (substitute it back or check it another way) "
                         "and correct any mistake you find."),
]
# Validation by INDEPENDENT RE-DERIVATION (pilot, 2026-10-08: a math-specialised 1.5B model does not follow
# "end with VERDICT: ..." meta-instructions; it re-solves instead). So it is asked to do exactly that: re-solve,
# box its own answer; verdict = whether it matches the candidate. An explicit VERDICT line is honoured if written.
VALIDATE_TEMPLATE = ("A previous solution to the problem below concluded that the final answer is {ans}.\n"
                     "Excerpt of that solution:\n{sol}\n\nVerify it: solve the problem again yourself, carefully and "
                     "independently, recomputing every step, and put YOUR final answer in \\boxed{{}}.\n\n"
                     "Problem:\n{q}")
# Synthesis: memory first, problem LAST, so the natural continuation is solving it (the pilot model echoed a
# long memory-after-problem prompt instead of answering).
SYNTH_TEMPLATE = ("Memory of {k} earlier attempts at the problem below:\n{memory}\n\n{selection}\n\n"
                  "Using this memory, solve the problem carefully and write the final answer in \\boxed{{}}.\n\n"
                  "Problem:\n{q}")
VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(CORRECT|INCORRECT)", re.I)


def mpv_attempt_question(question, k):
    return f"{question}\n\nApproach: {PERSPECTIVES[k][1]}"


def parse_verdict(text):
    found = VERDICT_RE.findall(str(text))
    return found[-1].upper() if found else None


def validation_verdict(text, candidate_answer, fmt="policy"):
    """Returns (verdict or None, how): an explicit VERDICT line if present, otherwise whether the validator's own
    re-derived boxed answer matches the candidate's answer. No gold answer is involved."""
    explicit = parse_verdict(text)
    if explicit:
        return explicit, "explicit"
    if candidate_answer is None:
        return None, "no_candidate_answer"
    own = extract_final(text, fmt)[1]
    if own is None:
        return None, "no_rederived_answer"
    return ("CORRECT" if answers_equivalent(own, candidate_answer) else "INCORRECT"), "rederived"


def answers_equivalent(a, b):
    """Candidate-vs-candidate equivalence (no gold involved): exact rational, string, or symbolic."""
    if a is None or b is None:
        return False
    ra, rb = rational(a), rational(b)
    if ra is not None and rb is not None:
        return ra == rb
    if str(a).strip() == str(b).strip():
        return True
    if parse is None:
        return False
    try:
        pa, pb = _mv_parse(str(a)), _mv_parse(str(b))
        return bool(pa and pb and verify(pa, pb, timeout_seconds=_to()) and verify(pb, pa, timeout_seconds=_to()))
    except MV_ERRORS:
        return False


def group_answers(answers):
    """Group equivalent answers; returns [{'answer', 'members': [idx...]}] in first-seen order."""
    groups = []
    for i, a in enumerate(answers):
        if a is None:
            continue
        for g in groups:
            if answers_equivalent(a, g["answer"]):
                g["members"].append(i)
                break
        else:
            groups.append({"answer": a, "members": [i]})
    return groups


def mpv_select(answers, verdicts):
    """Pick the answer group with the most (votes + passed validations); ties -> earliest attempt.
    Returns (selected_index or None, majority_index or None, groups)."""
    groups = group_answers(answers)
    if not groups:
        return None, None, groups
    majority = max(groups, key=lambda g: (len(g["members"]), -g["members"][0]))
    scored = max(groups, key=lambda g: (len(g["members"]) + sum(verdicts[i] == "CORRECT" for i in g["members"]),
                                        -g["members"][0]))
    return scored["members"][0], majority["members"][0], groups

# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def wilson(k, n, z=1.959963984540054):
    assert n > 0 and 0 <= k <= n
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(a_only, b_only):
    n = a_only + b_only
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / (2 ** n)) if n else 1.0


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def paired_bootstrap(a, b, draws=2000, seed=42):
    """95% CI of mean(a-b) resampling items (a, b: per-item 0/1 or probabilities)."""
    import numpy as np
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.ptp(d) == 0:                      # Hoeffding fallback for constant samples
        r = 2 * math.sqrt(math.log(2 / 0.05) / (2 * len(d)))
        return float(d.mean()), max(-1.0, d.mean() - r), min(1.0, d.mean() + r)
    rng = np.random.default_rng(seed)
    s = d[rng.integers(0, len(d), size=(draws, len(d)))].mean(axis=1)
    lo, hi = np.quantile(s, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)

# --------------------------------------------------------------------------------------
# Self-check (run before any GPU work)
# --------------------------------------------------------------------------------------

def self_check():
    B = "\\boxed"
    # strict numeric scoring
    assert score_text("<short>" + B + "{4.0}</short>", "4", "gsm8k")["correct"]
    assert not score_text("<short>" + B + "{47/3}</short>", "15", "gsm8k")["correct"]
    assert not score_text("<short>Earlier 503. Final " + B + "{10}</short>", "503", "aime2026")["correct"]
    assert score_text("<short>" + B + "{1,248}</short>", "1248", "gsm8k")["correct"]
    assert score_text("<short>" + B + "{\\$18}</short>", "18", "gsm8k")["correct"]
    # refusals and wrong dates never pass
    assert not score_text("<short>I'm here to help with general knowledge.</short>", "393", "aime2026")["correct"]
    assert not score_text("<short>On Wednesday, March 7, 2024 ...</short>", "1248", "gsm8k")["correct"]
    # modes
    assert parse_policy_output("<think>a</think> so " + B + "{3}")[0] == "think"
    assert parse_policy_output("<short>" + B + "{3}</short>")[0] == "short"
    assert parse_policy_output("<think>unfinished reasoning 3")[0] == "malformed"
    assert parse_policy_output("<short>" + B + "{3}</short> extra <short>4</short>")[0] == "malformed"
    assert parse_policy_output("<think>a</think>")[0] == "malformed"           # no answer after think
    assert not score_text("<think>so it is " + B + "{3}", "3", "gsm8k")["correct"]   # truncated think
    assert score_text("<think>work</think>The answer is " + B + "{3}.", "3", "gsm8k")["correct"]
    # answer inside think but none after it does not count
    assert not score_text("<think>" + B + "{3}</think>The end.", "3", "gsm8k")["correct"]
    # plain models (base / R1-distill)
    assert score_text("reasoning </think> final " + B + "{12}", "12", "gsm8k", fmt="plain")["correct"]
    assert score_text("Step 1 ... " + B + "{12}", "12", "gsm8k", fmt="plain")["correct"]
    assert not score_text("<think> long thinking without end " + B + "{12}", "12", "gsm8k", fmt="plain")["correct"]
    # format
    assert format_ok("<short>x " + B + "{1}</short>") and format_ok("<think>w</think> " + B + "{1}")
    assert not format_ok("<short>no box</short>") and not format_ok("plain " + B + "{1}")
    # repetition
    assert has_repetition_loop("<think>" + "the answer is " * 40 + "</think>")
    assert not has_repetition_loop("<think>2 + 3 = 5 so the total is 5</think>")
    # mode decision for think-rates: unfinished thinking counts as thinking
    assert chose_think("think", "<think>a</think> " + B + "{1}") and chose_think("malformed", "<think>cut off at the limit")
    assert not chose_think("short", "<short>" + B + "{1}</short>") and not chose_think("malformed", "<short>never closed")
    assert not chose_think("untagged", "plain answer " + B + "{1}")                        # base model: never thinks
    assert chose_think("untagged", "reasoning cut off", template_opens_think=True)          # AdaptThink, truncated
    assert not chose_think("short", "</think>\n\n" + B + "{1}", template_opens_think=True)  # AdaptThink NoThinking
    # a long normal solution that reuses phrases is NOT a loop (v1's 4-gram rule flagged these)
    steps = " ".join(f"Step {i}: the total number of apples is {3 * i}." for i in range(40))
    assert not has_repetition_loop("<think>" + steps + "</think> " + B + "{117}")
    assert has_repetition_loop("<think>" + steps + " So x = 1. Then" * 12)                  # mid-length loop
    assert has_repetition_loop("Let me compute. " + "We get 10^{3} mod 7 = 6, " * 200)          # long loop, tail check
    # rewards
    comps = ["<short>" + B + "{5}</short>", "<think>long</think> " + B + "{5}", "<short>" + B + "{6}</short>",
             "<think>never closed " + B + "{5}"]
    gold, task = ["5"] * 4, ["gsm8k"] * 4
    assert correctness_reward(None, comps, gold, task) == [1.0, 1.0, 0.0, 0.0]
    ml = mode_length_reward(None, comps, gold, task, completion_ids=[[0] * 50, [0] * 1500, [0] * 10, [0] * 2048])
    assert ml[0] > ml[1] > 0 and ml[2] == 0 and ml[3] == 0
    assert format_reward(comps) == [0.1, 0.1, 0.1, -0.5]
    assert think_rate(comps) == [0.0, 1.0, 0.0, 1.0]                     # cut-off thinking is still a think decision
    assert think_correct(None, comps, gold, task) == [0.0, 1.0, 0.0, 0.0]
    assert policy_think_prob(comps, policy_think_prob=[.2] * 4) == [.2] * 4 and policy_think_prob(comps) == [None] * 4
    # difficulty-aware mode reward: EASY group (4/4 correct) favours short; HARD group (1/4) favours think
    S, T = "<short>" + B + "{5}</short>", "<think>w</think> " + B + "{5}"
    W = "<short>" + B + "{6}</short>"
    easy = [S, S, T, T]
    hard = [T, W, W, W]
    ar = adaptive_mode_reward(None, easy + hard, ["5"] * 8, ["gsm8k"] * 8, id=["e"] * 4 + ["h"] * 4,
                              completion_ids=[[0] * n for n in (50, 100, 500, 1000, 800, 10, 10, 10)])
    assert abs(ar[0] - (MODE_BETA * 1.0 + MODE_LEN_ALPHA * 1.0)) < 1e-9          # shortest correct short, easy
    assert abs(ar[1] - MODE_BETA * 1.0) < 1e-9 and abs(ar[2] - MODE_LEN_ALPHA) < 1e-9 and ar[3] == 0.0
    assert abs(ar[4] - (MODE_BETA * 0.75 + MODE_LEN_ALPHA * 0.5)) < 1e-9 and ar[5:] == [0.0, 0.0, 0.0]
    assert max(ar) <= MODE_BETA + MODE_LEN_ALPHA <= LENGTH_ALPHA + 1e-12        # same bonus budget as R3
    assert group_keys(3, prompts=["a", "a", "b"]) == ["a", "a", "b"] and group_keys(2) == [0, 0]
    assert all_correct_group(None, easy + hard, ["5"] * 8, ["gsm8k"] * 8, id=["e"] * 4 + ["h"] * 4) == [1.0] * 4 + [0.0] * 4
    # published hybrid baselines (Thinkless / AdaptThink output formats)
    assert extract_final("<short>\nso " + B + "{2}", "hybrid")[:2] == ("short", "2")                     # Thinkless short
    assert extract_final("<think>\nhmm</think>\nso " + B + "{2}", "hybrid")[:2] == ("think", "2")        # Thinkless think
    assert extract_final("</think>\n\nDirect: " + B + "{7}", "hybrid")[:2] == ("short", "7")             # AdaptThink NoThinking
    assert extract_final("Let me think... </think> " + B + "{7}", "hybrid")[:2] == ("think", "7")      # AdaptThink Thinking
    assert extract_final("<think>\nnever closed " + B + "{7}", "hybrid")[0] == "malformed"
    # judge parsers
    assert parse_judge_v1(" 1.00") == 1.0 and parse_judge_v1("<think>score 0.3</think> SCORE: 0.9") == 0.9
    assert parse_judge_strict("The answer is 503. SCORE: 0.00") is None
    assert parse_judge_strict("SCORE: 0.75") == 0.75
    # real pilot outputs: the v1 parser reads a number from the judge's own re-solution (34 → clipped 1.0)
    real = " 0.00\n</think>\n\nTo simplify, sqrt(80) = 4 sqrt 5 ... \\boxed{\\dfrac{\\sqrt{179}}{6}}."
    assert parse_judge_v1(real) == 1.0 and parse_judge_fixed(real) == 0.0     # v1 read "80" from the re-solution
    assert parse_judge_fixed(" 13535\n</think>\n\nTo find the greatest integer ...") is None      # an answer, not a score
    assert parse_judge_fixed(" 70.00\n</think>\n\nGeometrically ...") is None
    assert parse_judge_fixed(" 0.67\nThe candidate is partly right. SCORE: 1.00") == 0.67           # completion wins
    assert parse_judge_fixed("Okay, 34 pages...</think>She reads \\boxed{8.5} per day.\n\nSCORE: 0.00") == 0.0
    assert parse_judge_fixed("Okay, let me check 12 and 7 ... (truncated, no verdict)") is None
    assert parse_judge_fixed("<think>2+2=4</think>The candidate is right.\nSCORE: 1.00") == 1.0
    # statistics
    assert abs(mcnemar_exact(3, 5) - 0.7265625) < 1e-12
    lo, hi = wilson(30, 30)
    assert hi > 0.999 and lo < 0.9
    assert all(abs(x - y) < 1e-12 for x, y in zip(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
    # MPV helpers
    assert parse_verdict("... so it holds. VERDICT: INCORRECT") == "INCORRECT"
    assert parse_verdict("**VERDICT: correct**") == "CORRECT" and parse_verdict("no verdict") is None
    sel, maj, groups = mpv_select(["5", "7", "5.0", None], ["INCORRECT", "CORRECT", "INCORRECT", None])
    assert maj == 0 and len(groups) == 2                       # "5" has 2 votes
    assert sel == 0                                            # 2 votes + 0 checks > 1 vote + 1 check
    sel, maj, _ = mpv_select(["5", "7"], ["INCORRECT", "CORRECT"])
    assert maj == 0 and sel == 1                               # tie on votes broken by validation
    assert validation_verdict("I recomputed: total = " + B + "{5}", "5.0") == ("CORRECT", "rederived")
    assert validation_verdict("<short>Recomputing gives " + B + "{7}</short>", "5") == ("INCORRECT", "rederived")
    assert validation_verdict("... VERDICT: INCORRECT", "5") == ("INCORRECT", "explicit")
    assert validation_verdict("I could not finish", "5") == (None, "no_rederived_answer")
    assert validation_verdict(B + "{5}", None) == (None, "no_candidate_answer")
    assert SYNTH_TEMPLATE.format(k=4, memory="m", selection="s", q="Q?").endswith("Problem:\nQ?")
    if parse is not None:   # symbolic checks need math-verify
        assert score_text("<short>" + B + "{2\\sqrt{2}}</short>", "\\sqrt{8}", "math500")["correct"]
        assert not score_text("<short>" + B + "{1.414214}</short>", "\\sqrt{2}", "math500")["correct"]
        assert score_text("<think>w</think> " + B + "{\\frac{1}{2}}", "0.5", "math500")["correct"]
        assert score_text("<short>" + B + "{18\\text{ apples}}</short>", "18", "gsm8k")["correct"]
    return True

# --------------------------------------------------------------------------------------
# Library-version safety: keep only kwargs the installed class accepts, and say what was dropped.
# --------------------------------------------------------------------------------------

def supported_kwargs(cls, kwargs, required=()):
    import dataclasses, inspect
    if dataclasses.is_dataclass(cls):
        names = {f.name for f in dataclasses.fields(cls)}
    else:
        names = set(inspect.signature(cls).parameters)
    kept = {k: v for k, v in kwargs.items() if k in names}
    dropped = sorted(set(kwargs) - set(kept))
    missing = [r for r in required if r not in kept]
    if missing:
        raise RuntimeError(f"{cls.__name__} lacks required options {missing}; check the library version.")
    if dropped:
        print(f"[compat] {cls.__name__} does not support {dropped}; they were NOT applied.")
    return kept