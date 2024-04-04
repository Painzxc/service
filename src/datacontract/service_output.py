from typing import List

import pydantic


class Ship(pydantic.BaseModel):
    """Датаконтракт выхода сервиса"""

    width: int = pydantic.Field(default=640)
    """Ширина преобразованного изображения"""
    height: int = pydantic.Field(default=480)
    """Высота преобразованного изображения"""
    channels: int = pydantic.Field(default=3)
    """Число каналов преобразованного изображения"""


class Plane(pydantic.BaseModel):
    """Датаконтракт выхода сервиса"""

    width: int = pydantic.Field(default=640)
    """Ширина преобразованного изображения"""
    height: int = pydantic.Field(default=480)
    """Высота преобразованного изображения"""
    channels: int = pydantic.Field(default=3)
    """Число каналов преобразованного изображения"""


class ServiceOutput(pydantic.BaseModel):
    """Датаконтракт выхода сервиса"""

    ships: List[Ship | None]
    planes: List[Plane | None]
