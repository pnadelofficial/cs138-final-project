# Air Traffic Control as a Deep Reinforcement Learning Problem
Find in this directory the code used to train and evaluate the BlueSky-Gym envrinoments.
* `recurrent_ppo.py`: A close, though simplified reimplementation of `sb3_contrib`'s Recurrent PPO, with thorough commenting and documention. 
* `testbed.py`: The abstraction and script used to coordinate training, saving and evaluation of models
* `test.sh`: The SLURM script used to run the above file on the Tufts HPC
* `requirements.txt`: The pip requirements to run the code
* `final_project_slides.pdf`: Slides summarizing our results
* `final_proejct_report.pdf`: The write-up of our report

To run any code, be aware that you must install the dependencies with: `pip install -r requirements.txt`. 