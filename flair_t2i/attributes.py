"""The seven attribute classes FLAIR routes, per spec section 3.1."""

from enum import Enum


class AttributeClass(str, Enum):
    IDENTITY = "identity"
    COLOR = "color"
    SIZE = "size"
    LIGHTING = "lighting"
    TEXTURE = "texture"
    STYLE = "style"
    ACTION = "action"


#: The four attributes used for the controllability curve (spec section 4)
#: and the FLUX mini-BASM (spec section 3.7).
CORE_ATTRIBUTES: tuple[AttributeClass, ...] = (
    AttributeClass.IDENTITY,
    AttributeClass.COLOR,
    AttributeClass.SIZE,
    AttributeClass.LIGHTING,
)
