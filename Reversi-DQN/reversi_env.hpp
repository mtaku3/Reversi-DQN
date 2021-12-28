#pragma once

#include <vector>
#include <creversi.h>

using namespace std;

vector<int> GenerateLegalMoveList(__Board bd) {
	__LegalMoveList lml = __LegalMoveList(bd);
	vector<int> legalMoves(lml.size());
	for (int i = 0; i < legalMoves.size(); i++) {
		legalMoves[i] = lml.next();
	}
	return legalMoves;
}

class ReversiEnv {
public:
	__Board board;

	bool is_agent_black;

	struct QData {
		__Board board;
		double reward;
		bool done;
	};

	ReversiEnv(bool is_agent_black) {
		board = __Board();

		this->is_agent_black = is_agent_black;
	}

	__Board reset() {
		board.reset();

		return board;
	}

	struct QData step(int move) {
		board.move(move);

		auto done = board.is_game_over();
		double reward;
		if (done) {
			reward = 0 < board.diff_num() ? 1.0 : (board.diff_num() < 0 ? -1.0 : 0.0);
			if (!is_agent_black) reward *= -1;
		}
		else {
			reward = 0;
		}

		return { board, reward, done };
	}
};