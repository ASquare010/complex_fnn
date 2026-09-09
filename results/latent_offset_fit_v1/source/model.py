"""Controlled affine subfamilies; identical arithmetic with one fixed zero buffer."""

from results.latent_activation_v1.source.model import make_model as original_model

FORMS = ("plain", "plain_first_lr", "latent_gain", "latent_offset", "latent_offset_first_lr",
         "latent_affine", "narrow_swiglu", "narrow_gelu", "full_swiglu", "full_gelu")
COUNTS = dict(zip(FORMS, (350208, 350208, 350232, 350232, 350232, 350256,
                        350208, 350208, 1179648, 1179648)))
HIDDEN = dict(zip(FORMS, (2048, 2048, 2048, 2048, 2048, 2048, 304, 456, 1024, 1536)))


def make_model(form, seed):
    if form == "plain_first_lr":
        model = original_model("plain", seed)
    elif form in ("latent_gain", "latent_offset", "latent_offset_first_lr"):
        model = original_model("latent_affine", seed)
        fixed = "theta_b" if form == "latent_gain" else "theta_a"
        for projection in (model.up, model.gate, model.down):
            zero = getattr(projection.curve, fixed).detach().clone()
            delattr(projection.curve, fixed)
            projection.curve.register_buffer(fixed, zero)
    else:
        model = original_model(form, seed)
    assert sum(p.numel() for p in model.parameters()) == COUNTS[form]
    return model
