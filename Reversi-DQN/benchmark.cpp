#include <iostream>

#include "DQNAgent.hpp"
#include "reversi_env.hpp"

using namespace std;

void print_board(__Board bd) {
	string str = bd.to_s_ffo();
	cout << " |abcdefgh" << endl << "-+--------" << endl;
	for (int i = 0; i < 8; i++) {
		cout << i + 1 << "|";
		for (int j = 0; j < 8; j++) {
			cout << str[i * 8 + j];
		}
		cout << endl;
	}
}

int main(void) {
	const bool is_agent_black = true;

	ReversiEnv env(is_agent_black);

	const int n_episodes = 50;

	DQNAgent agent(false);
	agent.loadParams();

	__Board current_state;
	int win = 0, lose = 0, draw = 0;
	for (int e = 0; e < n_episodes; e++) {
		current_state = env.reset();

		int action;
		struct ReversiEnv::QData _qdata;
		__Board next_state;
		double reward;
		bool done;
		while (1) {
			if (env.board.turn() == is_agent_black) {
				// AGENT
				action = agent.compute_action(current_state, false);
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

			if (done) {
				if (0 < reward) win++;
				else if (reward < 0) lose++;
				else draw++;
				print_board(env.board);
				break;
			}
			current_state = next_state;
		}
		cout << "EPISODE " << e + 1 << " END: " << (0 < reward ? "WIN" : (reward < 0 ? "LOSE" : "DRAW")) << endl;
	}

	cout << "WIN:" << win << " LOSE:" << lose << " DRAW:" << draw << endl;
	cout << "WINRATE:" << (double)win / n_episodes << endl;
}