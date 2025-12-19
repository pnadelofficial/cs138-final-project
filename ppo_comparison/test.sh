/cluster/tufts/tuftsai/pnadel01/cs138-final-project/scripts/
#!/bin/bash -l 

#SBATCH -J PPOComparison
#SBATCH --time=02-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -n 16
#SBATCH --mem=32g 
#SBATCH --output=PPOComparison%j.%N.out
#SBATCH --error=PPOComparison.%j.%N.err
#SBATCH --mail-type=ALL   
#SBATCH --mail-user=peter.nadel@tufts.edu

echo "Starting"
date

echo "Module loading"
module load anaconda/2023.07.tuftsai

echo "Activating env"
source activate sb3_rl

echo "Starting script"
python /cluster/tufts/tuftsai/pnadel01/cs138-final-project/scripts/ppo_comparison.py

echo "Script finished"