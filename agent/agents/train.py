import ray
from ray import air, tune
from agents.network.attention_net import AttentionWrapper
from agents.configs import get_configs
from ray.rllib.models import ModelCatalog


def main():
    ray.init(dashboard_host="0.0.0.0")
    ModelCatalog.register_custom_model("custom_model", AttentionWrapper)
    configs_dict = get_configs()
    
    stop = {
        "training_iteration": 50_000,
    }
    metric = "episodes_this_iter"

    t = tune.Tuner(
        "PPO",
        param_space=configs_dict,
        run_config=air.RunConfig(stop=stop,
        verbose=0,
        checkpoint_config=air.CheckpointConfig(num_to_keep=3,
                                               checkpoint_score_attribute=metric,
                                               checkpoint_score_order="min",
                                               checkpoint_frequency=50,
                                               checkpoint_at_end=True))
    )
    results = t.fit()

    best_result = results.get_best_result(metric=metric, mode="min", scope="all")

    print(
        "Best trial's lowest episode length (over all "
        "iterations): {}".format(best_result)
    )

    ray.shutdown()