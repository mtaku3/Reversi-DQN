#pragma once

#include <vector>
#include <creversi.h>

using namespace std;

vector<int> GenerateLegalMoveList(__Board bd) {
	__LegalMoveList lml = __LegalMoveList(bd);
	vector<int> legalMoves(lml.size());
	for (auto value : legalMoves) {
		value = lml.next();
	}
	return legalMoves;
}

class ReversiEnv {
public:
	__Board board;

	struct QData {
		__Board board;
		double reward;
		bool done;
	};

	ReversiEnv() {
		board = __Board();
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
			reward = board.diff_num() < 0 ? 1.0 : (board.diff_num() > 0 ? -1.0 : 0.0);
		}
		else {
			reward = 0;
		}

		return { board, reward, done };
	}
};