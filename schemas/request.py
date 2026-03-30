from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PredictionRequestBase64(BaseModel):

    image_base64: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded image data (without data URI prefix).",
        json_schema_extra={"example": "/9j/4AAQSkZJRgABAQ..."},
    )
    filename: Optional[str] = Field(
        default=None,
        description="Optional original filename for file type validation.",
        json_schema_extra={"example": "durian_sample.jpg"},
    )

    @field_validator("image_base64")
    @classmethod
    def validate_base64_not_empty(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("image_base64 must not be empty or whitespace.")
        if "," in stripped and stripped.startswith("data:"):
            return stripped.split(",", 1)[1]
        return stripped

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            return v.strip().lower()
        return v
