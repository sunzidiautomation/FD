"""Deterministic prompt parsing into attribute components (spec section 3.2).

A dependency parse binds modifiers to the head noun. Each modifier is
classified into one of the seven attribute classes by lexicon lookup.
Unknown modifiers are ignored rather than guessed at -- the calibration
prompts (spec section 3.4) are authored against this lexicon.
"""

from __future__ import annotations

from .attributes import AttributeClass
from .components import Component

ATTRIBUTE_LEXICON: dict[AttributeClass, frozenset[str]] = {
    AttributeClass.COLOR: frozenset(
        {
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "grey", "gray", "brown", "silver", "golden",
            "crimson", "teal", "navy", "beige",
        }
    ),
    AttributeClass.SIZE: frozenset(
        {
            "small", "large", "big", "tiny", "huge", "miniature", "massive",
            "compact", "enormous", "petite", "oversized",
        }
    ),
    AttributeClass.TEXTURE: frozenset(
        {
            "rusty", "smooth", "rough", "glossy", "matte", "shiny", "worn",
            "polished", "weathered", "cracked", "woven", "furry", "metallic",
        }
    ),
    AttributeClass.STYLE: frozenset(
        {
            "cyberpunk", "vintage", "retro", "futuristic", "minimalist",
            "baroque", "impressionist", "cartoon", "photorealistic",
            "watercolor", "sketch",
        }
    ),
    AttributeClass.LIGHTING: frozenset(
        {
            "warm", "cool", "bright", "dim", "golden", "moody", "harsh",
            "soft", "backlit", "sunlit", "neon", "candlelit",
        }
    ),
}

#: Scene-level attributes keep their own phrase rather than binding to the
#: object's head noun.
_SCENE_LEVEL = (AttributeClass.LIGHTING, AttributeClass.STYLE)

HEDGE_WORDS: frozenset[str] = frozenset(
    {
        "very", "extremely", "super", "highly", "intensely",
        "quite", "fairly", "moderately", "rather",
        "slightly", "somewhat", "mildly", "faintly", "barely",
        "not",
    }
)

_LIGHTING_HEADS = frozenset({"light", "lighting", "sunlight", "glow", "illumination"})


def _classify(token) -> AttributeClass | None:
    word = token.lemma_.lower()
    for attr, vocab in ATTRIBUTE_LEXICON.items():
        if word in vocab:
            return attr
    return None


def _hedge_for(token) -> str | None:
    for child in token.children:
        if child.lemma_.lower() in HEDGE_WORDS:
            return child.lemma_.lower()
    if token.head is not token and token.head.lemma_.lower() in HEDGE_WORDS:
        return token.head.lemma_.lower()
    return None


def _head_noun_chunk(doc):
    """The first noun chunk that is not part of a lighting/scene phrase."""
    for chunk in doc.noun_chunks:
        if chunk.root.lemma_.lower() in _LIGHTING_HEADS:
            continue
        words = [t.text for t in chunk if not t.is_stop or t.pos_ == "NOUN"]
        stripped = [t.text for t in chunk if _classify(t) is None and not t.is_stop]
        return chunk, " ".join(stripped) if stripped else " ".join(words)
    return None, ""


def parse_prompt(prompt: str, nlp=None) -> list[Component]:
    """Parse ``prompt`` into at most one Component per attribute class."""
    if nlp is None:  # pragma: no cover - convenience path
        import spacy

        nlp = spacy.load("en_core_web_sm")

    doc = nlp(prompt)
    chunk, identity_text = _head_noun_chunk(doc)

    found: dict[AttributeClass, Component] = {}

    if identity_text:
        found[AttributeClass.IDENTITY] = Component(
            id=f"c_{AttributeClass.IDENTITY.value}",
            text=identity_text,
            attr=AttributeClass.IDENTITY,
        )

    for token in doc:
        attr = _classify(token)
        if attr is None or attr in found:
            continue

        if attr in _SCENE_LEVEL:
            phrase = (
                " ".join(
                    t.text
                    for t in token.head.subtree
                    if t.lemma_.lower() not in HEDGE_WORDS
                )
                if token.head is not token
                else token.text
            )
            text = phrase.strip() or token.text
        else:
            text = f"{token.text} {identity_text}".strip()

        found[attr] = Component(
            id=f"c_{attr.value}",
            text=text,
            attr=attr,
            hedge=_hedge_for(token),
        )

    for token in doc:
        if AttributeClass.ACTION in found:
            break
        if token.pos_ == "VERB" and token.lemma_.lower() not in {"be"}:
            found[AttributeClass.ACTION] = Component(
                id=f"c_{AttributeClass.ACTION.value}",
                text=f"{identity_text} {token.text}".strip(),
                attr=AttributeClass.ACTION,
                hedge=_hedge_for(token),
            )

    order = list(AttributeClass)
    return [found[a] for a in order if a in found]
