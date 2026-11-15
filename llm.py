"""Tiny OpenAI-compatible client with retries + on-disk cache (cache/<hash>.json)."""
import hashlib, json, os, random, re, time, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

_client = None
_lock = threading.Lock()
os.makedirs("cache", exist_ok=True)
if os.path.exists(".env"):  # KEY=VALUE lines; never committed
    for ln in open(".env"):
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.strip().split("=", 1)
            os.environ.setdefault(k, v.strip().strip('"\''))


BACKEND = "adaption" if os.environ.get("ADAPTION_API_KEY") else "openai"


def client():
    global _client
    if _client is None:
        if BACKEND == "adaption":
            from adaption import Adaption
            _client = Adaption(max_retries=0, timeout=180)
        else:
            _client = OpenAI(base_url=os.environ["BASE_URL"], api_key=os.environ["API_KEY"], max_retries=0, timeout=120)
    return _client


def _call(model, messages, temperature, max_tokens):
    if BACKEND == "adaption":
        sys_ = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        inp = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        kw = dict(model=model, input=inp, temperature=temperature, max_output_tokens=max_tokens, store=False)
        if sys_:
            kw["instructions"] = sys_
        r = client().responses.create(**kw)
        if r.status not in ("completed", "incomplete"):
            raise RuntimeError(f"status {r.status}: {r.error}")
        return r.output_text or "", getattr(r.usage, "cost", None)
    r = client().chat.completions.create(model=model, messages=messages,
                                         temperature=temperature, max_tokens=max_tokens)
    return r.choices[0].message.content or "", None


def chat(model, messages, temperature=0.0, max_tokens=600, tag=""):
    key = hashlib.sha256(json.dumps([model, messages, temperature, max_tokens, tag]).encode()).hexdigest()[:24]
    path = f"cache/{key}.json"
    if os.path.exists(path):
        return json.load(open(path))["text"]
    for attempt in range(10):
        try:
            text, cost = _call(model, messages, temperature, max_tokens)
            json.dump({"model": model, "messages": messages, "text": text, "cost": cost, "ts": time.time()}, open(path, "w"))
            return text
        except Exception as e:
            msg = str(e)
            m = re.search(r"(?:try again in|retry after) (?:(\d+)m)?([\d.]+) ?s", msg)
            wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5) if m else min(60, 2 ** attempt + random.random())
            if attempt == 9 or ("401" in msg) or ("404" in msg and "model" in msg.lower()):
                print("FAIL", model, msg[:300], flush=True)
                return None
            time.sleep(wait)


def pmap(fn, xs, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


def spent():
    """Total $ cost recorded in cache/ (calls made before cost logging count as 0)."""
    import glob
    return sum(json.load(open(p)).get("cost") or 0 for p in glob.glob("cache/*.json"))
