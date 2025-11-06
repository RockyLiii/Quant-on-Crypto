import pydantic
import pydantic.alias_generators


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra="allow",
        populate_by_name=True,
        alias_generator=pydantic.alias_generators.to_camel,
    )
