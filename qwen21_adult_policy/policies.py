"""The ONLY place adult text lives in this package, and the whole runtime difference it makes.

Pure Python: no ComfyUI imports, so every rule here is unit-testable off a GPU and off a server. `nodes.py` imports
from here; this file imports only the policy engine beside it.

WHAT THIS PRODUCT IS. Erotic and fantasy images for adults, made by the person who asks for them, on Qwen Image
2.1. The form shows two gates, ✅ Consent and 🔞 18+, and with either one off nothing samples. Behind them, two rules
that no toggle overrides and that fail closed:

  1. Nobody under 18. Asked of the words (the request as typed, and again after Qwen's rewriter, when it is on), of
     every photo the person adds, and of the finished image before it is saved. An error, a missing answer or an
     unparseable answer counts as a stop.
  2. No famous real person named.

An explicit edit of a real photo runs on ✅ Consent, the person's own attestation that anyone real in it agreed. So
this package carries ONE policy for every mode, and the engine's `unlock` mechanism is deliberately unused: a face
match against a trained character is a later safeguard, and a half-wired unlock would be a ceiling nobody could see.
"""
import importlib
import pathlib
import re
import sys


def _engine():
    """The policy engine, `policy.py`, generated INTO this pack at build time, so the ordinary relative import is
    right and cannot pick up a stranger's copy. The by-path fallback is for the suite, which loads this file directly."""
    try:
        return importlib.import_module("." + "policy", __package__)
    except (ImportError, TypeError, SystemError):
        path = pathlib.Path(__file__).resolve().with_name("policy.py")
        if not path.is_file():
            raise ImportError("qwen21_adult_policy/policy.py is missing. It is generated from the shared engine; "
                              "reinstall the package.")
        spec = importlib.util.spec_from_file_location("qwen21_adult_policy_engine", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, mod)       # @dataclass resolves cls.__module__ through it
        spec.loader.exec_module(mod)
        return mod


_p = _engine()
Fact, Policy, Product = _p.Fact, _p.Policy, _p.Product
AGE_FACT_QUESTION = _p.AGE_FACT_QUESTION
INPUT_SCOPE_ALL = _p.INPUT_SCOPE_ALL
REFUSAL_STOPS_ALL_FLOWS = _p.REFUSAL_STOPS_ALL_FLOWS
FACT_SYSTEM = _p.FACT_SYSTEM
parse_answer = _p.parse_answer
clear_keys = _p.clear_keys
verdict = _p.verdict

# --------------------------------------------------------------------------- the facts
# The same wording as the brand's other products, so every ErosCraft product judges a request alike. The two text
# facts were measured on 2026-09-11 (0 false negatives and 0 false positives over a 12-prompt set, one batched call
# matching per-fact asking); the age question is the engine's own constant, measured the same day (6 of 6 minors
# caught including the 16 to 17 band, one false stop on an 18-year-old, so the error falls on the safe side).
#
# UNMEASURED ON THIS MODEL. Those numbers were measured with a Qwen3-VL 4B; here the questions go to the 8B that
# Qwen Image 2.1 loads as its text encoder. It is the same family and a larger model, which is a reason to expect
# no worse, not a measurement. The GPU pass records what it sees, on adult subjects only.
MINOR_TEXT = Fact("minor",
                  "is the main subject a child, teenager, or otherwise under 18? "
                  "(Any adult, including an elderly person, is NOT under 18.)")
FAMOUS_TEXT = Fact("well_known_real_person",
                   "does the request name a specific famous or well-known real person "
                   "(a celebrity, public figure or influencer)?")
AGE_IMAGE = Fact("anyone_under_18", AGE_FACT_QUESTION)

# --------------------------------------------------------------------------- what the person reads
# Every sentence a stop can show. `nodes.py` picks among them; the DECISION to stop is verdict()'s alone and is never
# revisited by the choice of words.
CONSENT_OFF = ("Turn on ✅ Consent to run. It says that everyone shown is an adult and that any real person in "
               "a photo agreed to this.")
ADULT_OFF = "Turn on 🔞 18+ to run. It says that you are 18 or over."
BOTH_GATES_OFF = ("Turn on ✅ Consent and 🔞 18+ to run. Consent says that everyone shown is an adult and that any "
                  "real person in a photo agreed to this; 18+ says that you are 18 or over.")

STOPS = {
    "toggle_consent": CONSENT_OFF,
    "toggle_adult": ADULT_OFF,
    "minor": "That asks for someone under 18, so nothing was made. ErosCraft is adults only, and this check cannot "
             "be turned off.",
    "anyone_under_18": "Someone in a photo you added reads as under 18, so nothing was made. ErosCraft is adults "
                       "only, and this check cannot be turned off.",
    "well_known_real_person": "That names a famous real person, so nothing was made. ErosCraft does not make "
                              "pictures of public figures. Describe an invented person instead.",
}

# The words the prompt enhancer wrote are checked too, and a stop there says so: the person did not type them.
REWRITE_STOPS = {
    "minor": "The prompt enhancer's rewrite describes someone under 18, so nothing was made. Run again with the "
             "enhancer off, or say the age plainly: ErosCraft is adults only.",
    "well_known_real_person": "The prompt enhancer's rewrite names a famous real person, so nothing was made. Run "
                              "again with the enhancer off, or describe an invented person.",
}

# The check before saving asks the age question of the finished image. The image WAS made, so its sentence says so.
OUTPUT_STOPS = {
    "anyone_under_18": "The finished image reads as someone under 18, so it was not saved. ErosCraft is adults "
                       "only, and this check cannot be turned off. Say the age plainly in the prompt (\"a woman in "
                       "her late twenties\") and run again.",
}

# A check that could not answer still stops the run, because every check fails closed; what it may not do is tell
# the person they asked for a minor. So a stop with no clear answer behind it says what actually happened.
UNCLEAR = ("The safety check could not get a clear answer, so nothing was made. That is not a finding about your "
           "request: run it again, and if it stops again, the console line says why.")
OUTPUT_UNCLEAR = ("The safety check could not get a clear answer about the finished image, so it was not saved. "
                  "That is not a finding about the image: run it again, and if it stops again, the console line "
                  "says why.")


def gates_off(toggle_state) -> str:
    """The one sentence for the gates that are off, in the form's order, or "" when both are on."""
    off = tuple(name for name in ("consent", "adult") if not toggle_state.get(name))
    return {(): "", ("consent",): CONSENT_OFF, ("adult",): ADULT_OFF}.get(off, BOTH_GATES_OFF)


# --------------------------------------------------------------------------- the policy
# ONE policy, every mode. `unlock` is deliberately unset (see the module docstring), and nothing here asks a fact
# keyed "explicit", so the engine's face-match branch is unreachable. The amendment and the negative are empty: this
# graph has no rewriter amendment to append to, and it samples at CFG 1, where a negative prompt does nothing.
EROS = Policy(
    name="EROS",
    text_facts=(MINOR_TEXT, FAMOUS_TEXT),
    input_facts=(AGE_IMAGE,),
    input_scope=INPUT_SCOPE_ALL,          # every photo, not only the first
    image_facts=(AGE_IMAGE,),             # the finished image
    stops=STOPS,
    amendment="",
    negative="",
    refusal_stops=REFUSAL_STOPS_ALL_FLOWS,
)


def product(consent, adult):
    """The PRODUCT the form's two toggles build, and the whole of what the checks are told."""
    return Product(
        name="ErosCraft",
        by_flow={},
        default=EROS,
        toggles=("consent", "adult"),
        toggle_state={"consent": bool(consent), "adult": bool(adult)},
        # Every machine of this brand mounts one shared store, so each product keeps its own folder.
        output_root="ErosCraft/qwen21",
        engines=(),
        edit_engine=None,
        civitai={"host": "https://civitai.red", "nsfw": True},
    )


# --------------------------------------------------------------------------- the ➕ picker
# What the Civitai Red picker lists and fetches. Adult resources ARE the point of the picker; these are what it
# never offers, checked on the server at search time AND again at download time, because a page can be made to ask
# for anything. The engine's own flags (a real person's likeness, a minor, SFW-only) are checked by civitai.py; these
# words catch what an uploader did not flag. The list errs toward refusing: a resource wrongly hidden costs a
# search, and one wrongly offered costs the rule.
REFUSED_WORDS = ("loli", "shota", "child", "children", "kid", "kids", "underage", "under 18", "teen", "teenager",
                 "schoolgirl", "school girl", "schoolboy", "young girl", "young boy", "little girl", "little boy",
                 "age slider", "ageslider", "celebrity", "celeb", "lookalike", "look-alike", "likeness",
                 "non-consensual", "nonconsensual", "rape")
_REFUSED_RE = re.compile(r"(?<![a-z])(%s)(?![a-z])" % "|".join(re.escape(w) for w in REFUSED_WORDS), re.I)


def refused_words(*texts):
    """The first refused word in any of `texts` (a model's name, its tags, a version's name), or None."""
    for t in texts:
        m = _REFUSED_RE.search(str(t or ""))
        if m:
            return m.group(1).lower()
    return None


# The resources the picker shows first, before any search. Each carries its author's sampling, applied on the
# ◆ and ★ tiers (the turbo runs its own schedule): sampler|scheduler|cfg.
RECOMMENDED = (
    {"versionId": 3351951, "modelId": 2958918, "name": "NSFW LORA | Qwen Image 2.1", "creator": "TheseAlpacas",
     "sampling": "er_sde|beta|1.0",
     "why": "The most downloaded adult LoRA for Qwen Image 2.1. Its author asks for sampler er_sde and scheduler "
            "beta, which are applied for you on the ◆ and ★ tiers."},
)
SAMPLING_BY_VERSION = {r["versionId"]: r["sampling"] for r in RECOMMENDED}
