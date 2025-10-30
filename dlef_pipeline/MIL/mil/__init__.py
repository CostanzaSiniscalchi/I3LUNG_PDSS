from ._params import (
    mil_config,
    TrainerConfig,
    MILModelConfig
)

from .train import (
    train_mil,
    _train_multimodal_mixed_mil,
    build_fastai_learner,
    
)
from .eval import (
    eval_mil,
    predict_mil,
    predict_from_mixed_bags
)

#from .utils import load_model_weights, load_mil_config
"""from ._registry import (
    list_trainers, list_models, is_trainer, is_model,
    get_trainer, get_model, get_model_config_class,
    build_model_config, register_trainer, register_model,
)"""

from ._registry import (
    get_trainer, 
    get_model,
    build_model_config, 
    register_trainer, 
    register_model,
)

# -----------------------------------------------------------------------------

@register_trainer
def fastai():
    return TrainerConfig

# -----------------------------------------------------------------------------

@register_model
def mb_attention_mil():
    from .models import MultiModal_Mixed_Attention_MIL
    return MultiModal_Mixed_Attention_MIL