from enum import Enum


class AttributeClass(str, Enum):
    IDENTITY = "identity"
    COLOR = "color"
    SIZE = "size"
    LIGHTING = "lighting"
    TEXTURE = "texture"
    STYLE = "style"
    ACTION = "action"


#: The four attributes used for the controllability curve

CORE_ATTRIBUTES: tuple[AttributeClass, ...] = (
    AttributeClass.IDENTITY,
    AttributeClass.COLOR,
    AttributeClass.SIZE,
    AttributeClass.LIGHTING,
)
