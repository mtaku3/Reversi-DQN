#include <iostream>

#include "DQNAgent.hpp"
#include "reversi_env.hpp"

using namespace std;

int main(void) {
	const bool is_agent_black = true;

	ReversiEnv env(is_agent_black);

	int n_episodes = 2;

	DQNAgent agent;
	
	int total_steps = 0;
	int n_add_episodes = n_episodes;
	int overall_win = 0, overall_lose = 0, overall_draw = 0;
	int cur_win = 0, cur_lose = 0, cur_draw = 0;
	__Board current_state;
	for (int e = 0; e < n_episodes; e++) {
		current_state = env.reset();
		
		int action;
		struct ReversiEnv::QData _qdata;
		__Board next_state;
		double reward;
		bool done;
		while(1) {
			total_steps++;

			if (env.board.turn() == is_agent_black) {
				// AGENT
				action = agent.compute_action(current_state);
			}
			else {
				// OPPONENT
				vector<int> legalMoves = GenerateLegalMoveList(env.board);
				action = legalMoves[rand_int(0, legalMoves.size() - 1)];
			}
			_qdata = env.step(action);
			next_state = _qdata.board;
			reward = _qdata.reward;
			done = _qdata.done;

			if (env.board.turn() == is_agent_black) {
				// ONLY ON AGENT TURN
				agent.store_episode(current_state, action, reward, next_state, done);
			}

			if (done) {
				if (reward < 0) {
					overall_win++;
					cur_win++;
				}
				else if (0 < reward) {
					overall_lose++;
					cur_lose++;
				}
				else {
					overall_draw++;
					cur_draw++;
				}
				agent.update_exploration_probability();
				break;
			}
			current_state = next_state;
		}

		if (agent.batch_size <= total_steps) {
			cout << "current episode = " << e + 1 << endl;
			agent.train();
		}

		if (e + 1 == n_episodes) {
			cout << "--- TEMPORARY TRAIN PAUSE ---" << endl;
			cout << "OVERALL | WIN:" << overall_win << " LOSE:" << overall_lose << " DRAW:" << overall_draw << " WINRATE:" << (double)overall_win / n_episodes << endl;
			cout << "CURRENT | WIn:" << cur_win << " LOSE:" << cur_lose << " DRAW:" << cur_draw << " WINRATE:" << (double)cur_win / n_add_episodes << endl;
			cout << "TOTAL STEPS:" << total_steps << endl;
			cout << "Train more episodes : "; cin >> n_add_episodes;
			n_episodes += n_add_episodes;
			cur_win = cur_lose = cur_draw = 0;
		}
	}

	agent.model.save("reversi-weights", content_type::weights, file_format::json);
}