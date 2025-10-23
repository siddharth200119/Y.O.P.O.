from pydantic import BaseModel, model_validator
from typing import List, Literal, Optional

classes = Literal[
    'button', 'link', 'input_field', 'checkbox', 'radio_button',
    'dropdown', 'slider', 'toggle', 'label', 'text_block', 'icon',
    'menu_item', 'text_area', 'select_menu', 'clickable_region'
]

classes_map: List[str] = [
    'button', 'link', 'input_field', 'checkbox', 'radio_button',
    'dropdown', 'slider', 'toggle', 'label', 'text_block', 'icon',
    'menu_item', 'text_area', 'select_menu', 'clickable_region'
]

class BBOX(BaseModel):
    xyxy: List[float]
    xywh: List[float]

class Result(BaseModel):
    bbox: BBOX
    class_idx: int
    class_name: Optional[classes] = None
    conf: float
    content: Optional[str]

    @model_validator(mode="after")
    def assign_class_name(cls, model):
        if model.class_name is None:
            idx = model.class_idx
            if 0 <= idx < len(classes_map):
                model.class_name = classes_map[idx]
        return model
