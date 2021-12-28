import os
import gym
import torch
import numpy as np
import creversi.gym_reversi
from creversi import creversi

from DQNAgent import DQNAgent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on " + str(device))

env = gym.make("Reversi-v0").unwrapped

n_episodes = 10

agent = DQNAgent()

total_steps = 0;
n_add_episodes = n_episodes;
stats = {
	"overall": {
		"win": 0,
		"lose": 0,
		"draw": 0
	},
	"current_episode": {
		"win": 0,
		"lose": 0,
		"draw": 0
	}
}
while 0 < n_add_episodes:
	for e in range(n_episodes - n_add_episodes, n_episodes):
		current_state = env.reset();
	
		while True:
			total_steps += 1

			if env.board.turn == True:
				# AGENT
				action = agent.compute_action(current_state);
			else:
				# OPPONENT
				legalMoves = list(env.board.legal_moves)
				action = np.random.choice(legalMoves)
			next_state, reward, done, _ = env.step(action)

			if env.board.turn == True:
				# ONLY ON AGENT TURN
				agent.store_episode(current_state, action, reward, next_state, done);

			if done:
				if 0 < reward:
					stats["overall"]["win"] += 1
					stats["current_episode"]["win"] += 1
				elif reward < 0:
					stats["overall"]["lose"] += 1
					stats["current_episode"]["lose"] += 1
				else:
					stats["overall"]["draw"] += 1
					stats["current_episode"]["draw"] += 1
				agent.update_exploration_probability();
				break;
			current_state = next_state;

		if agent.batch_size <= total_steps:
			agent.train()
		print('Episode %d end: %s (exploration_proba = %lf)' % (e + 1, ("Win" if reward < 0 else "Lose" if 0 < reward else "Draw"), agent.exploration_proba))

		if (e + 1 == n_episodes):
			print("--- TEMPORARY TRAIN PAUSE ---\n")
			print("OVERALL | WIN:" + str(stats["overall"]["win"]) + " LOSE:" + str(stats["overall"]["lose"]) + " DRAW:" + str(stats["overall"]["draw"]) + " WINRATE:" + str(stats["overall"]["win"] / n_episodes) + "\n")
			print("CURRENT | WIn:" + str(stats["current_episode"]["win"]) + " LOSE:" + str(stats["current_episode"]["lose"]) + " DRAW:" + str(stats["current_episode"]["draw"]) + " WINRATE:" + str(stats["current_episode"]["win"] / n_add_episodes) + "\n")
			print("TOTAL STEPS:" + str(total_steps) + "\n")
			print("Train more episodes : ")
			n_add_episodes = int(input())
			n_episodes += n_add_episodes
			stats["current_episode"]["win"] = 0
			stats["current_episode"]["lose"] = 0
			stats["current_episode"]["draw"] = 0

# Save model weights
nets = {
	1: [ "conv1", "bn1", "conv2", "bn2", "conv3", "bn3", "conv4", "bn4", "conv5", "bn5", "conv6", "bn6", "conv7", "bn7", "conv8", "bn8", "conv9", "bn9", "conv10", "bn10" ],
	2: [ "fcl1", "fcl2" ]
}

def dump_layer_weights(f, weight, bias):
	for w in weight:
		f.write('%.24f ' % w)
	_ = f.write('\n')
	for b in bias:
		f.write('%.24f ' % b)
	_ = f.write('\n')



def dump_net_weights(output_folder):
	for net_id in nets:
		layers = nets[net_id]
		output_file_path = os.path.join(output_folder, '%02d.weights' % net_id)
		
		print('Saving weights to %s' % output_file_path)
		with open(output_file_path, 'w') as f:
			for layer in layers:
				weight = agent.model._modules[layer].weight
				bias = agent.model._modules[layer].bias
				dump_layer_weights(f, torch.flatten(weight), torch.flatten(bias))

dump_net_weights("../Reversi-DQN")