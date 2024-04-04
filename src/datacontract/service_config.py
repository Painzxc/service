import pydantic


class ServiceConfig(pydantic.BaseModel):
    """_summary_

    Args:
        pydantic (_type_): _description_

    Returns:
        _type_: _description_
    """

    # target_width: int
    # target_height: int

    name_of_classifier: str
    path_to_classifier: str
    name_of_detector: str
    path_to_detector: str
