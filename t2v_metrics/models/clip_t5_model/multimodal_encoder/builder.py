import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg):
    """

    Parameters
    ----------
    vision_tower_cfg
    kwargs

    Returns
    -------

    """

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        vision_tower_model = CLIPVisionTower(args=vision_tower_cfg)
        vision_tower_model.load_model(model_name=vision_tower)
        return vision_tower_model
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
