from agents.env_wrapper.curriculum_env import CurriculumSimulatorEnv

from ray.rllib.env.apis.task_settable_env import TaskSettableEnv, TaskType
from ray.rllib.env.env_context import EnvContext

ITER_VAL = 0
NUM_TRAIN_WEEKS_PCT = 0.1

def curriculum_fn(
    train_results: dict, task_settable_env: TaskSettableEnv, env_ctx: EnvContext
) -> TaskType:
    
    global ITER_VAL
    global NUM_TRAIN_WEEKS_PCT

    if NUM_TRAIN_WEEKS_PCT < 1:
        ITER_VAL += 1

        if ITER_VAL % 10 == 0:
            NUM_TRAIN_WEEKS_PCT += 0.001
            ITER_VAL = 0

    return NUM_TRAIN_WEEKS_PCT


def base_configs():
    stock_env = CurriculumSimulatorEnv
    return {
        "env": stock_env,
        "framework": "torch",
        "num_cpus_for_driver": 2,
        "gamma": 0.99,
        "num_gpus": 1,
        "num_workers": 30,
        "vf_share_layers": True,
        "vf_loss_coeff": 4e-5,  # https://github.com/ray-project/ray/issues/5278
    }


def env_train_configs():
    return {
        "env_config": {
            "num_weeks_train": -1,
            "show_trade_result": False,
            "evaluation": False,
        },
        "env_task_fn": curriculum_fn,
    }


def env_evaluation_configs():
    return {
        "evaluation_interval": 1000,
        "evaluation_num_episodes": 5,
        "evaluation_config": {
            "exploration_fraction": 0,
            "exploration_final_eps": 0,
            "exploration": False,
            "env_config": {
                "num_weeks_train": -1,
                "show_trade_result": True,
                "evaluation": True,
            },
        },
    }


def trainer_configs():
    return {
        "train_batch_size": 22 * 42 * 30,
        "sgd_minibatch_size": 1024,  # 32 * NumWorkers
        "shuffle_sequences": False,
        "batch_mode": "complete_episodes",
    }


def model_configs():

    m_conf = {
        "model": {
            "custom_model": "custom_model",
        }
    }

    custom_model_conf = {
        "use_attention": True,
        "max_seq_len": 42,
        "attention_num_transformer_units": 6,
        "attention_num_heads": 4,
        "attention_dim": 128,
        "attention_memory_inference": 50,
        "attention_memory_training": 50,
        "attention_head_dim": 128,
        "attention_position_wise_mlp_dim": 256,
        "attention_init_gru_gate_bias": 2.0,
    }

    m_conf["model"]["custom_model_config"] = custom_model_conf

    return m_conf


def get_configs():
    configs = dict(
        base_configs(),
        **env_train_configs(),
        **trainer_configs(),
        **model_configs(),
    )
    return configs
