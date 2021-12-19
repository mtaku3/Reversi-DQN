#include <iostream>

#include "DQNAgent.hpp"
#include "reversi_env.hpp"

int main(void) {
	ReversiEnv env;

	DQNAgent agent;

	agent.compute_action(env.board);
}