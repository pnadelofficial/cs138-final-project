import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO
from recurrent_ppo import RecurrentPPO

bluesky_gym.register_envs()

class TestBed:
    def __init__(self, env_name, timesteps=1e6):
        self.env_name = env_name
        self.timesteps = timesteps

        self.env = gym.make(env_name)
    
        self.two_layer_kwargs = {
            "lstm_hidden_size":256,
            "n_lstm_layers":2,
            "enable_critic_lstm":True
        }
        self.four_layer_kwargs = {
            "lstm_hidden_size":256,
            "n_lstm_layers":4,
            "enable_critic_lstm":True
        }

        self.standard_model = PPO("MultiInputPolicy", self.env, verbose=1, learning_rate=3e-4, tensorboard_log=f"./{self.env_name.lower()}_tensorboard_logs/")
        self.model_2layer = RecurrentPPO(policy="MultiInputLstmPolicy", env=self.env, verbose=1, learning_rate=5e-4, tensorboard_log=f"./{self.env_name.lower()}_tensorboard_logs/", policy_kwargs=self.two_layer_kwargs, n_epochs=17, n_steps=2048, ent_coef=0.02, batch_size=512)
        self.model_4layer = RecurrentPPO("MultiInputLstmPolicy", env=self.env, verbose=1, learning_rate=5e-4, tensorboard_log=f"./{self.env_name.lower()}_tensorboard_logs/", policy_kwargs=self.four_layer_kwargs, n_epochs=17, n_steps=2048, ent_coef=0.02, batch_size=512)
        self.models = {
            "PPO_2":self.standard_model,
            "RecurrentPPO_1":self.model_2layer,
            "RecurrentPPO_2":self.model_4layer
        }

    def _train_model(self, model, log_name):
        model.set_env(self.env)
        model.learn(total_timesteps=self.timesteps, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(f"{log_name.lower()}_{env_name.lower()}")
        env.close()
    
    def test(self):
        for log_name in self.models:
            self._train_model(self.models[log_name], log_name)
    
def main():
    env_names = [
        "SectorCREnv-v0",
        "DescentEnv-v0", 
        "HorizontalCREnv-v0", 
        "MergeEnv-v0", 
        "PlanWaypointEnv-v0", 
        "StaticObstacleEnv-v0", 
        "VerticalCREnv-v0"
    ]

    for env_name in env_names:
        test_bed = TestBed(env_name=env_name)
        test_bed.test()

if __name__ == "__main__":
    main()