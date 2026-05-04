# core/clip_labels.py

CLIP_LABEL_MAP: dict[str, str] = {
    "a raw, unedited, high-quality photograph of a real durian fruit, "
    "distinctly showing its sharp natural green-brown thorns or fresh yellow fleshy pods inside.":
        "Durian Asli",
    "a digital illustration, 3d render, vector graphic, cartoon, anime, "
    "painting, sketch, drawing, or ai-generated synthetic art of a fruit.":
        "Ilustrasi / Render 3D",
    "a digital screenshot, meme, promotional flyer, poster, graphic design, "
    "or an image containing visible text, words, icons, and UI elements.":
        "Tangkapan Layar / Teks",
    "a photograph prominently featuring a human face, a person, crowds, "
    "or visible human hands holding, opening, or interacting with objects.":
        "Manusia / Anggota Tubuh",
    "a photograph of cooked meals, yellow rice, curry, plated dishes on banana leaves, "
    "or durian-flavored desserts, ice cream, pastries, and cakes.":
        "Makanan Olahan / Hidangan",
    "a photograph of similar rough green fruits like jackfruit, breadfruit, or soursop, "
    "or just a random pile of green leaves, grass, tree branches, and plants.":
        "Buah Lain / Dedaunan",
    "a photograph of furry animals, pets, or spiky animals like hedgehogs or porcupines.":
        "Hewan",
    "a general photograph of everyday household items, electronics, indoor furniture, "
    "vehicles, buildings, or landscape scenery without a clear main subject.":
        "Objek Acak / Pemandangan",
}

# Index 0 selalu "Durian Asli" — digunakan sebagai acuan penolakan di CLIPService.
LABEL_PROMPTS: list[str] = list(CLIP_LABEL_MAP.keys())
LABEL_NAMES:   list[str] = list(CLIP_LABEL_MAP.values())

DURIAN_LABEL_INDEX = 0  # indeks label "Durian Asli" di LABEL_PROMPTS