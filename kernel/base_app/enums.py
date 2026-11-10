from enum import IntEnum, StrEnum


class GeneralStatusTypes(StrEnum):
    ACTIVE = "A"
    INACTIVE = "I"
    ARCHIVED = "Z"

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


class ActionEnum(StrEnum):
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"

    @classmethod
    def choices(cls):
        return [(key.value, key.name.capitalize()) for key in cls]
