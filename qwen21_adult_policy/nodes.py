"""The two gates a person sees, the three checks behind them, and the ➕ Civitai Red picker.

The policy itself is `policies.py`. This file is the wiring: it asks the questions through the Qwen3-VL 8B that
Qwen Image 2.1 already loads as its text encoder, reads the answers fail-closed, and stops.

WHY THE QUESTIONS GO THROUGH THAT MODEL. A separate classifier would be a second model in memory, a second thing to
install and a second thing to drift. The stock encoder is a full Qwen3-VL 8B, language head included (read from the
file's own header on 2026-09-24), so asking it costs one short generation per question set and nothing else. It is
ALWAYS the stock encoder, loaded as its own input: the form lets a person pick another encoder for the picture, and
no form control may change how a background rule is judged.

EVERY ASK PRINTS ONE LINE: what was asked, how many pictures went with it, how long the answer was, whether the
thinking block closed, and the parsed answers. A stop a person cannot trace back to a question is a stop they will
assume is a bug.
"""
import json
import os

from .policies import (FACT_SYSTEM, OUTPUT_STOPS, OUTPUT_UNCLEAR, RECOMMENDED, REWRITE_STOPS, SAMPLING_BY_VERSION,
                       UNCLEAR, clear_keys, gates_off, parse_answer, product, refused_words, verdict)

CATEGORY = "Qwen Image 2.1/ErosCraft"
NONE = "(none)"
THINK_START, THINK_END = "<think>", "</think>"
PHOTO_SLOTS = 10
# The longest side a photo is asked about at. The question is "is anyone in this under 18", which a 768-pixel view
# answers as well as a 2K one, and ten 2K photos would cost the checker ten times the vision tokens for nothing.
CHECK_SIDE = 768


def _shrink(image):
    """One photo, RGB, its longest side at most CHECK_SIDE, as a [1, H, W, 3] tensor."""
    import torch
    im = image[:1, :, :, :3]
    h, w = int(im.shape[1]), int(im.shape[2])
    scale = CHECK_SIDE / max(h, w)
    if scale >= 1:
        return im
    size = (max(28, int(h * scale)), max(28, int(w * scale)))
    return torch.nn.functional.interpolate(im.movedim(-1, 1), size=size, mode="bilinear",
                                           align_corners=False).movedim(1, -1)


def _generate(clip, system_prompt, prompt, max_new_tokens, images=()):
    """One greedy, thinking-on generation through the stock encoder. Returns the RAW text, think block included.

    Raw, because the fail-closed rule is ours to apply: an answer whose thinking never closed is no answer at all.
    The photos go in as a LIST, one vision slot each, so photos of different sizes are all seen as they are."""
    imgs = [_shrink(i) for i in images]
    pads = "<|vision_start|><|image_pad|><|vision_end|>" * len(imgs)
    # No think block is opened here: the model opens one if it reasons, and _answer reads a reply either way. Forcing
    # one open on a model that answers directly would leave every reply "unclosed", and a gate that refuses every
    # input protects nobody (the brand's MiniMax H3 product measured exactly that failure on 2026-09-19).
    text = ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s%s<|im_end|>\n<|im_start|>assistant\n"
            % (system_prompt, pads, prompt))
    tokens = clip.tokenize(text, images=imgs, skip_template=True, min_length=1)
    ids = clip.generate(tokens, do_sample=False, max_length=int(max_new_tokens))
    return clip.decode(ids)


# The suite substitutes this with a fake, which is how every fail-closed path is tested off a GPU.
GENERATE = _generate


def _answer(raw):
    """The model's answer, or None when there is no usable one.

      * the thinking block CLOSED: the answer is the text after it;
      * a block OPENED and never closed: the generation ran out of room mid-thought, which is not a "no", so it is
        None, and `parse_answer` turns None into the unsafe value for every fact;
      * NO thinking block at all: the model answered directly, and that answer is the answer.
    """
    if raw is None:
        return None
    text = str(raw)
    if THINK_END in text:
        return text.split(THINK_END, 1)[1]      # a second close tag stays in the answer, which makes it unclear
    if THINK_START in text:
        return None
    return text


def _merge(into, new, facts):
    """Fold one ask's answers into the running set: once any source says unsafe, it stays unsafe."""
    unsafe = {f.key: f.unsafe for f in facts}
    for key, value in new.items():
        if key not in into or value == unsafe.get(key, True):
            into[key] = value
    return into


def _clear_keys(answer, facts):
    """The facts the model answered with a plain yes or no. Only these can name a rule in a stop."""
    return clear_keys(answer, facts)


def _stop_message(policy, facts, clearly_unsafe, stops, unclear):
    """The sentence for a stop verdict() has ALREADY made. Choosing words cannot turn a stop into a pass."""
    known = {f.key: f for f in policy.text_facts + policy.input_facts + policy.image_facts}
    for key, value in facts.items():
        fact = known.get(key)
        if fact is not None and value == fact.unsafe and key in clearly_unsafe:
            return stops.get(key) or policy.stops.get(key, "Stopped: %s." % key)
    return unclear


def ask(facts, body, clip, max_new_tokens, images=(), where="", clearly_unsafe=None):
    """Ask one set of yes/no questions in a single call, and read the reply fail-closed."""
    if not facts:
        return {}
    prompt = body.rstrip() + "\n\n" + "\n".join("%s: %s" % (f.key, f.question) for f in facts)
    raw, failed = None, ""
    try:
        raw = GENERATE(clip, FACT_SYSTEM, prompt, max_new_tokens, images=tuple(images))
    except Exception as exc:                        # noqa: BLE001 - any failure is a stop, never a pass
        failed = "%s: %s" % (type(exc).__name__, exc)
    answer = _answer(raw)
    parsed = parse_answer(answer, facts)
    if clearly_unsafe is not None:
        unsafe = {f.key: f.unsafe for f in facts}
        clearly_unsafe.update(k for k in _clear_keys(answer, facts) if parsed[k] == unsafe[k])
    print("[ErosCraft %s] asked %s | pictures %d | raw %d chars | %s | %s%s" % (
        where or "gate", ",".join(f.key for f in facts), len(images), 0 if raw is None else len(str(raw)),
        "answered" if answer is not None else "no answer (a thinking block never closed): every answer is unsafe",
        parsed, "" if not failed else " | the checker raised %s" % failed))
    return parsed


def _photos(kw):
    """The photos the form sent, in slot order, skipping the empty ones."""
    return [kw["photo_%d" % i] for i in range(1, PHOTO_SLOTS + 1) if kw.get("photo_%d" % i) is not None]


class Qwen21AdultProduct:
    """✅ Consent and 🔞 18+, and the policy they unlock.

    Both are form rows, both ship off, and with either one off the run stops before anything samples. They are
    attestations: turning them on is the person's statement about themselves and about anyone real in a photo, and
    it moves responsibility to them. They do NOT override the two background rules, nobody under 18 and no famous
    real person named, which fail closed and cannot be turned off from the form or from anywhere else."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "consent": ("BOOLEAN", {
                "default": False, "label_on": "✅ confirmed", "label_off": "off",
                "tooltip": "Everyone shown is an adult, and any real person in a photo has agreed to this. Turning "
                           "this on is your statement, not a check."}),
            # label_on is not "🔞 18+": the row's own label already says that.
            "adult": ("BOOLEAN", {
                "default": False, "label_on": "✅ I am 18+", "label_off": "off",
                "tooltip": "You are 18 or over. This confirms YOUR age, not the age of anyone in a picture, which "
                           "is checked separately and cannot be turned off."}),
        }}

    RETURN_TYPES = ("PRODUCT",)
    RETURN_NAMES = ("product",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "ErosCraft's policy for this product: two gates, and behind them everything except minors and " \
                  "named real people."

    @classmethod
    def VALIDATE_INPUTS(cls, consent, adult):
        """Refuse at queue time, so a gate that is off costs no GPU and no wait."""
        pol = product(consent, adult)
        ok, message = verdict(pol.default, {}, True, dict(pol.toggle_state), False)
        return True if ok else (gates_off(pol.toggle_state) or message)

    def run(self, consent, adult):
        return (product(consent, adult),)


class Qwen21AdultGate:
    """The check before anything samples: the words and every photo.

    It sits ON the path rather than beside it: the request the enhancer and the encoder read comes out of this node,
    and so does every photo, so there is no way to reach the sampler around it. With the gate stopped its outputs
    never exist and nothing downstream runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "product": ("PRODUCT", {"tooltip": "The two gates and the policy behind them."}),
                "words": ("STRING", {"forceInput": True, "tooltip": "The request, as the form assembled it."}),
                "checker": ("CLIP", {"tooltip": "The stock Qwen3-VL 8B encoder the questions are asked through."}),
                "max_tokens": ("INT", {
                    "default": 4096, "min": 64, "max": 8192,
                    "tooltip": "Room for the checker to think and answer. Too small and the thinking never closes, "
                               "which counts as unsafe and stops the run."}),
            },
            "optional": {"photo_%d" % i: ("IMAGE",) for i in range(1, PHOTO_SLOTS + 1)},
        }

    RETURN_TYPES = ("STRING",) + ("IMAGE",) * PHOTO_SLOTS + ("POLICY",)
    RETURN_NAMES = ("words",) + tuple("photo_%d" % i for i in range(1, PHOTO_SLOTS + 1)) + ("policy",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Checks the words and every photo before anything samples. Passes them through unchanged."

    def run(self, product, words, checker, max_tokens, **photos):
        if product is None:
            raise RuntimeError("No policy is wired to this gate, so nothing was made. The 🔞 ErosCraft policy node "
                               "belongs on its `product` input.")
        policy = product.policy_for("")
        toggles = {name: bool(product.toggle_state.get(name)) for name in product.toggles}

        # The gates first, and on their own: a run stopped by a toggle must not cost a generation.
        ok, message = verdict(policy, {}, True, toggles, False)
        if not ok:
            message = gates_off(toggles) or message
            print("[ErosCraft gate] stopped by a gate before any model call: %s" % message)
            raise RuntimeError(message)

        facts, clearly_unsafe = {}, set()
        _merge(facts, ask(policy.text_facts, 'This is the request for an image: "%s"' % (words or ""),
                          checker, max_tokens, where="gate/words", clearly_unsafe=clearly_unsafe),
               policy.text_facts)
        pictures = _photos(photos)
        if pictures:
            # EVERY photo, in one call: each one is its own vision slot, so ten photos are ten looks, not two.
            _merge(facts, ask(policy.input_facts,
                              "Look at the %d attached photo%s." % (len(pictures), "" if len(pictures) == 1 else "s"),
                              checker, max_tokens, images=pictures, where="gate/photos",
                              clearly_unsafe=clearly_unsafe),
                   policy.input_facts)
        ok, message = verdict(policy, facts, True, toggles, False)
        if not ok:
            message = _stop_message(policy, facts, clearly_unsafe, policy.stops, UNCLEAR)
            print("[ErosCraft gate] stopped: %s" % message)
            raise RuntimeError(message)
        print("[ErosCraft gate] clear: %s" % facts)
        return (words,) + tuple(photos.get("photo_%d" % i) for i in range(1, PHOTO_SLOTS + 1)) + (policy,)


class Qwen21AdultRewriteCheck:
    """The words the prompt enhancer wrote, checked like the words the person typed.

    The enhancer is a language model, and what it writes is new text nobody typed. It goes to the encoder only
    through this node. With the enhancer off the two texts are the same and nothing is asked."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "policy": ("POLICY",),
            "checker": ("CLIP",),
            "before": ("STRING", {"forceInput": True, "tooltip": "The request before the rewrite."}),
            "after": ("STRING", {"forceInput": True, "tooltip": "The request after the rewrite."}),
            "prompt": ("STRING", {"forceInput": True, "tooltip": "The finished prompt, which passes through."}),
            "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, policy, checker, before, after, prompt, max_tokens):
        if policy is None:
            raise RuntimeError("The rewrite check has no policy wired to it, so nothing was made.")
        if (after or "").strip() == (before or "").strip():
            return (prompt,)
        clearly_unsafe = set()
        facts = ask(policy.text_facts, 'This is the request for an image: "%s"' % after, checker, max_tokens,
                    where="rewrite", clearly_unsafe=clearly_unsafe)
        ok, message = verdict(policy, facts, True, {}, False)
        if not ok:
            message = _stop_message(policy, facts, clearly_unsafe, REWRITE_STOPS, UNCLEAR)
            print("[ErosCraft rewrite] stopped: %s" % message)
            raise RuntimeError(message)
        print("[ErosCraft rewrite] clear: %s" % facts)
        return (prompt,)


class Qwen21AdultOutputCheck:
    """The check before an image is written: the age question, on what was actually made.

    It returns an ExecutionBlocker rather than raising, which is how ComfyUI stops an output node and everything
    downstream of it. A missing policy is a block, never a pass-through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "The finished image."}),
            "policy": ("POLICY", {"tooltip": "The policy the gate judged this run under."}),
            "checker": ("CLIP",),
            "max_tokens": ("INT", {"default": 4096, "min": 64, "max": 8192}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "Checks the finished image before it is saved."

    def run(self, image, policy, checker, max_tokens):
        if policy is None:
            message = ("The output check has no policy wired to it, so nothing was saved. Rebuild the workflow: "
                       "the gate's `policy` output belongs on this node.")
            print("[ErosCraft output] blocked: %s" % message)
            return (_blocker(message),)
        clearly_unsafe = set()
        # Every image of the batch, each its own slot; alpha is dropped by _shrink, since the check reads people.
        frames = [image[i:i + 1] for i in range(int(image.shape[0]))]
        facts = ask(policy.image_facts, "Look at the attached image%s, which was just made."
                    % ("" if len(frames) == 1 else "s"), checker, max_tokens, images=frames, where="output",
                    clearly_unsafe=clearly_unsafe)
        ok, message = verdict(policy, facts, True, {}, False)
        if ok:
            print("[ErosCraft output] clear: %s" % facts)
            return (image,)
        message = _stop_message(policy, facts, clearly_unsafe, OUTPUT_STOPS, OUTPUT_UNCLEAR)
        print("[ErosCraft output] blocked: %s" % message)
        return (_blocker(message),)


def _blocker(message):
    """ComfyUI's own way to stop an output node and everything below it."""
    try:
        from comfy_execution.graph_utils import ExecutionBlocker
    except Exception:
        try:
            from comfy_execution.graph import ExecutionBlocker
        except Exception:
            raise RuntimeError(message)
    return ExecutionBlocker(message)


# --------------------------------------------------------------------------- the ➕ picker
def picked_loras():
    """The LoRAs the picker has fetched, as the combo lists them: (none) first, then loras/Qwen/Civitai/*."""
    try:
        import folder_paths
        from .civitai import CIVITAI_DIR
        names = [n for n in folder_paths.get_filename_list("loras")
                 if n.replace("\\", "/").startswith(CIVITAI_DIR + "/")]
    except Exception:                               # the suite imports this file without ComfyUI
        names = []
    return [NONE] + sorted(names)


def lora_sampling(name):
    """The sampling a picked LoRA's author asks for, from the recommended table by its Civitai version, or ""."""
    if not name or name == NONE:
        return ""
    try:
        import folder_paths
        path = folder_paths.get_full_path("loras", name)
        with open(path + ".civitai.json", encoding="utf-8") as fh:
            vid = int(json.load(fh).get("modelVersionId") or 0)
    except Exception:
        return ""
    return SAMPLING_BY_VERSION.get(vid, "")


class Qwen21LoraPicker:
    """➕ a Civitai Red LoRA for Qwen Image 2.1, and how strongly it applies.

    The list is what the picker has fetched to this machine; the ➕ Browse Civitai Red button under the row searches
    and downloads with the machine's own token. `(none)` is a real pass-through, and there is no placeholder file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name": (picked_loras(), {"default": NONE,
                          "tooltip": "A LoRA fetched with ➕ Browse Civitai Red, or (none)."}),
            "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                   "tooltip": "How strongly the LoRA pulls. 1.0 is its author's intent; lower it "
                                              "if it overwhelms the picture."}),
        }}

    RETURN_TYPES = ("STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("lora_name", "strength", "sampling")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, lora_name, strength):
        return (lora_name, float(strength), lora_sampling(lora_name))


CIVITAI_SEARCH_LIMIT = 24


def civitai_search(query=None, cursor=None, host="https://civitai.red", limit=CIVITAI_SEARCH_LIMIT):
    """Qwen Image 2.1 LoRAs on Civitai Red, refused ones already removed. Returns (items, next_cursor).

    Civitai files Qwen Image 2.1 under TWO base models: "Qwen 2.1", and "Qwen 2" for uploads made before that tag
    existed (measured 2026-09-24). Both are asked for; a "Qwen 2" version is kept only when civitai_refusal says it is
    really 2.1. The query string is BUILT here from a fixed set of parameters; nothing the page sends becomes a URL."""
    import urllib.parse
    from . import civitai as cv
    params = [("types", "LORA"), ("baseModels", "Qwen 2.1"), ("baseModels", "Qwen 2"),
              ("sort", "Most Downloaded"), ("nsfw", "true"), ("limit", str(int(limit)))]
    if query:
        params.append(("query", str(query)[:200]))
    if cursor:
        params.append(("cursor", str(cursor)[:200]))
    url = cv.civitai_safe_url("%s/api/v1/models?%s" % (host.rstrip("/"), urllib.parse.urlencode(params)))
    headers = {}
    tok = cv.civitai_token()
    if tok:
        headers["Authorization"] = "Bearer %s" % tok
    with cv._fetch_validated(url, hosts=cv.CIVITAI_HOSTS, headers=headers or None, timeout=30) as r:
        data = json.load(r)
    items = []
    for model in data.get("items") or []:
        if refused_words(model.get("name"), " ".join(model.get("tags") or [])):
            continue
        flags = {k: model.get(k) for k in ("poi", "minor", "sfwOnly")}
        versions = []
        for v in model.get("modelVersions") or []:
            if cv.civitai_refusal({**v, "model": {**flags, "name": model.get("name")}}) or refused_words(v.get("name")):
                continue
            versions.append({"id": v.get("id"), "name": v.get("name"), "baseModel": v.get("baseModel"),
                             "trainedWords": v.get("trainedWords") or [],
                             "images": [{"url": im.get("url")} for im in (v.get("images") or [])[:1]
                                        if not im.get("poi") and not im.get("minor")]})
        if versions:
            items.append({"id": model.get("id"), "name": model.get("name"),
                          "creator": (model.get("creator") or {}).get("username"),
                          "downloadCount": (model.get("stats") or {}).get("downloadCount"),
                          "modelVersions": versions})
    return items, (data.get("metadata") or {}).get("nextCursor")


def civitai_fetch(version_id, host="https://civitai.red"):
    """Download one Civitai version to loras/Qwen/Civitai/, by id, with the server's own token. Returns the combo name.

    Two-step, because the storage refuses a request carrying an Authorization header: ask Civitai with the Bearer and
    do NOT follow the redirect, then fetch the signed target with no header, every hop validated."""
    import hashlib
    import urllib.error
    import folder_paths
    from . import civitai as cv
    vid = int(version_id or 0)
    if not vid:
        raise ValueError("no versionId")
    api_url = cv.civitai_safe_url("%s/api/v1/model-versions/%d" % (host.rstrip("/"), vid))
    with cv._fetch_validated(api_url, hosts=cv.CIVITAI_HOSTS, timeout=30) as r:
        version = json.load(r)
    refusal = cv.civitai_refusal(version) or (
        refused_words(version.get("name"), (version.get("model") or {}).get("name")) and "its name is refused here")
    if refusal:
        raise PermissionError("That LoRA is not available here: %s." % refusal)
    files = [f for f in (version.get("files") or []) if (f.get("name") or "").endswith(".safetensors")]
    primary = next((f for f in files if f.get("primary")), files[0] if files else None)
    if primary is None:
        raise ValueError("that version has no .safetensors file")
    tok = cv.civitai_token()
    if not tok:
        raise ValueError("no CIVITAI_TOKEN on the server: Civitai requires one to download. Set it and restart "
                         "ComfyUI.")
    url = cv.civitai_safe_url(primary.get("downloadUrl") or "%s/api/download/models/%d" % (host.rstrip("/"), vid))
    root = folder_paths.get_folder_paths("loras")[0]
    out_dir = os.path.join(root, *cv.CIVITAI_DIR.split("/"))
    os.makedirs(out_dir, exist_ok=True)
    name = cv.civitai_filename(vid, version.get("name") or (version.get("model") or {}).get("name"))
    dest = os.path.join(out_dir, name)
    try:
        src = cv._fetch_validated(url, hosts=cv.CIVITAI_HOSTS, redirect_hosts=None,
                                  headers={"Authorization": "Bearer %s" % tok}, timeout=120)
    except urllib.error.HTTPError as e:
        raise ValueError("Civitai answered %d: the token may lack access" % e.code) from None
    digest, written = hashlib.sha256(), 0
    with src, open(dest, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            digest.update(chunk)
            out.write(chunk)
    want = ((primary.get("hashes") or {}).get("SHA256") or "").lower()
    if want and digest.hexdigest() != want:
        os.remove(dest)
        raise ValueError("the download did not match Civitai's SHA256: nothing kept")
    with open(dest + ".civitai.json", "w", encoding="utf-8") as fh:
        json.dump({"air": version.get("air"), "modelVersionId": vid, "name": version.get("name"),
                   "baseModel": version.get("baseModel"), "trainedWords": version.get("trainedWords") or [],
                   "sha256": digest.hexdigest(), "bytes": written}, fh, indent=2)
    rel = "%s/%s" % (cv.CIVITAI_DIR, name)
    print("[ErosCraft picker] fetched %s (%.1f MB) from Civitai version %d" % (rel, written / 1e6, vid))
    return {"name": rel, "trainedWords": version.get("trainedWords") or [], "sampling": SAMPLING_BY_VERSION.get(vid, "")}


try:                                            # pragma: no cover, only inside a running server
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.post("/qwen21/civitai/search")
    async def _qwen21_civitai_search(request):
        """The ➕ picker's list, fetched by the server because the page's CSP forbids civitai.red."""
        try:
            body = await request.json()
            items, nxt = civitai_search(query=body.get("query"), cursor=body.get("cursor"))
            return web.json_response({"items": items, "nextCursor": nxt, "recommended": list(RECOMMENDED)})
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:
            return web.json_response({"error": "%s: %s" % (type(e).__name__, e)}, status=502)

    @PromptServer.instance.routes.post("/qwen21/civitai/fetch")
    async def _qwen21_civitai_fetch(request):
        """Download one version, by id, with the server's own token. Refused again here: the page is not the guard."""
        try:
            body = await request.json()
            return web.json_response(civitai_fetch(body.get("versionId")))
        except PermissionError as e:
            return web.json_response({"error": str(e)}, status=403)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        except Exception as e:                  # never take the server down over a download
            return web.json_response({"error": "%s: %s" % (type(e).__name__, e)}, status=500)
except Exception:                               # pragma: no cover: importable without ComfyUI
    pass


NODE_CLASS_MAPPINGS = {
    "Qwen21AdultProduct": Qwen21AdultProduct,
    "Qwen21AdultGate": Qwen21AdultGate,
    "Qwen21AdultRewriteCheck": Qwen21AdultRewriteCheck,
    "Qwen21AdultOutputCheck": Qwen21AdultOutputCheck,
    "Qwen21LoraPicker": Qwen21LoraPicker,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21AdultProduct": "🔞 ErosCraft policy",
    "Qwen21AdultGate": "🔞 ErosCraft gate",
    "Qwen21AdultRewriteCheck": "🔞 ErosCraft rewrite check",
    "Qwen21AdultOutputCheck": "🔞 ErosCraft output check",
    "Qwen21LoraPicker": "➕ Civitai Red LoRA",
}
