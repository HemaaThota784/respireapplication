"""Run once:  python inspect_models.py   -> prints input/output shapes of both models."""
import sys, keras
for p in sys.argv[1:] or ["models/respire_logits.keras", "models/xray_pneumonia_model.keras"]:
    m = keras.models.load_model(p, compile=False, safe_mode=False)
    print(p, "\n  input :", m.input_shape, "\n  output:", m.output_shape,
          "\n  last layer:", m.layers[-1].name, getattr(m.layers[-1], "activation", None))
