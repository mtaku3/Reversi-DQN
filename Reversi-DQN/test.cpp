#include <iostream>

#include "DQNAgent.hpp"
#include "reversi_env.hpp"

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

	DQNAgent agent(false);
	agent.loadParams();
	
	__Board current_state;
	int action;
	struct ReversiEnv::QData _qdata;
	__Board next_state;
	double reward;
	bool done;
	while (1) {
		if (env.board.turn() == is_agent_black) {
			// AGENT
			action = agent.compute_action(current_state, false);
			cout << "AGENT TOOK " << to_s(action) << endl << endl;
		}
		else {
			// OPPONENT
			print_board(env.board);
			
			vector<int> legalMoves = GenerateLegalMoveList(env.board);
			cout << "--- LEGAL MOVES ---" << endl;
			for (auto move : legalMoves) {
				cout << to_s(move) << endl;
			}
			cout << "-------------------" << endl;
			do {
				string _action;
				cin >> _action;
				action = to_hand(_action);
			} while (!env.board.is_legal(action));
		}
		_qdata = env.step(action);
		next_state = _qdata.board;
		reward = _qdata.reward;
		done = _qdata.done;

		if (done) {
			if (reward < 0) cout << "BLACK WIN";
			else if (0 < reward) cout << "WHITE WIN";
			else cout << "DRAW";
			cout << endl;
			break;
		}
		current_state = next_state;
	}
}