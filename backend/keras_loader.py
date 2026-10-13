"""Load .keras files saved by a NEWER Keras (e.g. Colab) with an older local Keras.

Newer Keras versions write extra config keys (VarianceScaling 'input_axes'/'output_axes',
BatchNormalization 'renorm*', ...). Older Keras rejects unknown keys. They are irrelevant for
inference (they affect initialisation / training only; weights come from the file), so on
failure we load a temporary copy of the model with those keys removed. If Keras names further
unknown keys in its error, they are removed too and loading is retried automatically.
"""
import json
import logging
import os
import re
import tempfile
import zipfile

import keras

log = logging.getLogger("respira")

_KNOWN_NEW_KEYS = {"input_axes", "output_axes", "renorm", "renorm_clipping", "renorm_momentum"}


def _strip(o, keys):
    if isinstance(o, dict):
        for k in keys:
            o.pop(k, None)
        for v in o.values():
            _strip(v, keys)
    elif isinstance(o, list):
        for v in o:
            _strip(v, keys)


def _patched_copy(path: str, keys) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".keras")
    os.close(fd)
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data)
                _strip(cfg, keys)
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(item, data)
    return tmp


def _keys_from_error(msg: str):
    keys = set(re.findall(r"unexpected keyword argument '(\w+)'", msg))
    for group in re.findall(r"Unrecognized keyword arguments[^{]*\{([^}]*)\}", msg):
        keys |= set(re.findall(r"'(\w+)'\s*:", group))
    return keys


def _load(p):
    return keras.models.load_model(p, compile=False, safe_mode=False)


def load_keras_model(path: str):
    try:
        return _load(path)
    except (TypeError, ValueError) as e:
        err = e
    log.warning("Standard load failed (%s...). Retrying with compatibility patch.", str(err)[:100])
    strip = set(_KNOWN_NEW_KEYS)
    for _ in range(8):
        tmp = _patched_copy(path, strip)
        try:
            model = _load(tmp)
            log.info("Model loaded after removing config keys: %s", sorted(strip))
            return model
        except (TypeError, ValueError) as e:
            err = e
            new = _keys_from_error(str(e)) - strip
            if not new:
                break
            strip |= new
        finally:
            os.remove(tmp)
    raise err
