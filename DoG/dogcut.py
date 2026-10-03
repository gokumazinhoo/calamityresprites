#!/usr/bin/env python3
"""
dogcut.py - recorta as 3 formas do Devourer of Gods em Cabeca / Corpo / Cauda / Jaw / Glow / MapIcon
com EXATAMENTE o tamanho dos PNGs originais (hitbox, origem e rotacao preservadas).

Uso:
  python dogcut.py --orig PASTA_ORIGINAIS --list
  python dogcut.py --orig PASTA_ORIGINAIS --art1 semarmadura1forma.png --art2 comarmadura2forma.png \
                   --art3 fantasmagorico3forma.jpg --out saida --preview
  (tire --preview para gerar so os PNGs finais)

PASTA_ORIGINAIS = pasta com os PNGs originais do DoG (a do seu repositorio, clonada).
O script descobre forma/parte pelo NOME do arquivo:
  DoGP1* -> forma 1 | DoGP2*Antimatter -> forma 3 (fantasma) | DoGP2* -> forma 2
  contem Head/Body/Tail/Jaw -> parte ; termina em Glow/Glowmask/MapIcon -> variante
"""
import argparse, os, re
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# ---------------- CONFIG (ajuste olhando o --preview) ----------------
# As artes tem a cabeca EMBAIXO; no jogo a frente fica EM CIMA -> giramos 180 graus.
ROT = 180
# Caixas em fracoes (x0,y0,x1,y1) da arte JA GIRADA (cabeca no topo, cauda embaixo).
BOXES = {
    1: {"Head": (0.00, 0.00, 1.00, 0.385), "Tail": (0.25, 0.862, 0.75, 1.00)},
    2: {"Head": (0.00, 0.00, 1.00, 0.255), "Tail": (0.00, 0.865, 1.00, 1.00)},
    3: {"Head": (0.00, 0.00, 1.00, 0.390), "Tail": (0.00, 0.860, 1.00, 1.00)},
}
# Corpo: faixa horizontal (x0,x1 em fracao) e posicao vertical (fracao) onde comeca o segmento.
BODY = {1: (0.32, 0.68, 0.55), 2: (0.04, 0.96, 0.46), 3: (0.10, 0.90, 0.50)}
BODY_SCALE_TO_WIDTH = True    # corpo e escalado para a largura do PNG original
# Fundo
BG_FORM = {1: ("black", 28), 2: ("color", 18), 3: ("black", 22)}
GLOW_SAT_MIN, GLOW_VAL_MIN = 0.45, 0.60
# ---------------------------------------------------------------------

def load_art(path, form):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    mode, tol = BG_FORM[form]
    if mode == "black":
        lum = a.max(axis=2)
        alpha = np.clip((lum - tol) * (255.0 / 40.0), 0, 255)   # fade suave no preto
        # mantem escuros internos: so apaga preto conectado a borda
        bg = lum <= tol
        lab, _ = ndimage.label(bg)
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        outside = np.isin(lab, list(edge))
        alpha = np.where(outside, 0, 255)
        # borda suave
        alpha = np.where(outside, 0, 255).astype(np.uint8)
    else:
        ref = a[2, 2]
        bg = (np.abs(a - ref).sum(axis=2) <= tol)
        lab, _ = ndimage.label(bg)
        edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        outside = np.isin(lab, list(edge))
        alpha = np.where(outside, 0, 255).astype(np.uint8)
    rgba = np.dstack([a.astype(np.uint8), alpha])
    out = Image.fromarray(rgba, "RGBA")
    return out.rotate(ROT, expand=True) if ROT else out

def crop_frac(img, box):
    w, h = img.size
    return img.crop((round(box[0]*w), round(box[1]*h), round(box[2]*w), round(box[3]*h)))

def fit(img, size, stretch=False):
    W, H = size
    if stretch:
        return img.resize((W, H), Image.NEAREST)
    bb = img.getbbox()
    if bb: img = img.crop(bb)
    s = min(W / img.width, H / img.height)
    nw, nh = max(1, round(img.width*s)), max(1, round(img.height*s))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    # cabeca alinhada ao topo (frente), cauda ao fundo
    canvas.paste(img, ((W-nw)//2, 0), img)
    return canvas

def fit_tail(img, size):
    W, H = size
    bb = img.getbbox()
    if bb: img = img.crop(bb)
    s = min(W / img.width, H / img.height)
    nw, nh = max(1, round(img.width*s)), max(1, round(img.height*s))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(img, ((W-nw)//2, 0), img)   # a ponta e a ultima; o encaixe com o corpo fica no topo
    return canvas

def make_body(art, form, size):
    W, H = size
    x0, x1, y0 = BODY[form]
    aw, ah = art.size
    cx0, cx1 = round(x0*aw), round(x1*aw)
    cw = cx1 - cx0
    s = W / cw
    ch = max(1, round(H / s))
    top = min(max(0, round(y0*ah)), ah - ch)
    seg = art.crop((cx0, top, cx1, top + ch)).resize((W, H), Image.NEAREST)
    # corpo e opaco: tapa buracos de alpha para os segmentos emendarem sem frestas
    a = np.asarray(seg).copy()
    return Image.fromarray(a, "RGBA")

def make_glow(img):
    a = np.asarray(img).astype(np.float32) / 255.0
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    mx, mn = a[..., :3].max(axis=2), a[..., :3].min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    m = (sat >= GLOW_SAT_MIN) & (mx >= GLOW_VAL_MIN) & (al > 0.5)
    out = a.copy(); out[..., 3] = np.where(m, al, 0)
    out[~m] = 0
    return Image.fromarray((out*255).astype(np.uint8), "RGBA")

def classify(fname):
    base = os.path.splitext(fname)[0]
    m = re.match(r"DoGP(\d)", base, re.I)
    if not m: return None
    form = int(m.group(1))
    if "antimatter" in base.lower(): form = 3
    low = base.lower()
    variant = "map" if "mapicon" in low else "glowmask" if "glowmask" in low else "glow" if "glow" in low else "main"
    part = next((p for p in ("Head", "Body", "Tail", "Jaw") if p.lower() in low), None)
    return form, part, variant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--art1"); ap.add_argument("--art2"); ap.add_argument("--art3")
    ap.add_argument("--out", default="saida")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.orig) if f.lower().endswith(".png"))
    if a.list:
        for f in files:
            print(f"{f:50s} {Image.open(os.path.join(a.orig, f)).size}  -> {classify(f)}")
        return

    arts = {}
    for k, p in ((1, a.art1), (2, a.art2), (3, a.art3)):
        if p: arts[k] = load_art(p, k)
    os.makedirs(a.out, exist_ok=True)

    if a.preview:
        for k, art in arts.items():
            pv = Image.new("RGBA", art.size, (40, 40, 40, 255)); pv.alpha_composite(art)
            d = ImageDraw.Draw(pv); w, h = art.size
            for part, bx in BOXES[k].items():
                d.rectangle([bx[0]*w, bx[1]*h, bx[2]*w-1, bx[3]*h-1], outline=(255, 255, 0, 255), width=2)
                d.text((bx[0]*w+4, bx[1]*h+4), part, fill=(255, 255, 0, 255))
            x0, x1, y0 = BODY[k]
            d.rectangle([x0*w, y0*h, x1*w, y0*h+80], outline=(0, 255, 0, 255), width=2)
            pv.save(os.path.join(a.out, f"_preview_forma{k}.png"))
        print("previews salvos em", a.out); return

    for f in files:
        c = classify(f)
        if not c: continue
        form, part, variant = c
        if form not in arts or part is None:
            if form in arts and variant == "map": pass
            else: continue
        size = Image.open(os.path.join(a.orig, f)).size
        art = arts.get(form)
        if art is None: continue
        if variant == "map" or part is None:
            head = fit(crop_frac(art, BOXES[form]["Head"]), size); out = head
        elif part == "Jaw":
            out = Image.new("RGBA", size, (0, 0, 0, 0))   # presas ja fazem parte da cabeca
        elif part == "Body":
            out = make_body(art, form, size)
        elif part == "Tail":
            out = fit_tail(crop_frac(art, BOXES[form]["Tail"]), size)
        else:
            out = fit(crop_frac(art, BOXES[form]["Head"]), size)
        if variant in ("glow", "glowmask"):
            out = make_glow(out)
        out.save(os.path.join(a.out, f))
        print("ok", f, size)

if __name__ == "__main__":
    main()
