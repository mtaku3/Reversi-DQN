#pragma once

#include <iostream>
#include <random>
#include <tiny_dnn/tiny_dnn.h>

#include "reversi_env.hpp"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

using namespace std;

random_device seed_gen;
mt19937 engine(seed_gen());

double rand_real(double min, double max) {
	uniform_real_distribution<> dist(min, max);

	return dist(engine);
}

int rand_int(int min, int max) {
	uniform_int_distribution<> dist(min, max);

	return dist(engine);
}

vector<vec_t> ConvertState(__Board board) {
	float data[2][64];
	board.piece_planes((char*)&data);
	vector<vec_t> res;
	
	for (int i = 0; i < 2; i++) {
		vec_t vec_tmp;
		for (int j = 0; j < 64; j++) {
			vec_tmp.push_back(data[i][j]);
		}
		res.push_back(vec_tmp);
	}

	return res;
}

class DQNAgent {
public:
	double lr;
	double gamma;
	double exploration_proba;
	double exploration_proba_decay;
	int batch_size;

	network<sequential> model;

	DQNAgent() {
		lr = 0.001;
		gamma = 0.99;
		exploration_proba = 1.0;
		exploration_proba_decay = 0.005;
		batch_size = 32;

		adam optimizer;

		model << convolutional_layer<relu>(8, 8, 3, 2, 2, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 2)
			<< convolutional_layer<relu>(8, 8, 3, 2, 2, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 2)
			<< convolutional_layer<relu>(8, 8, 3, 2, 2, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 2)
			<< convolutional_layer<relu>(8, 8, 3, 2, 2, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 2)
			<< convolutional_layer<relu>(8, 8, 3, 2, 2, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 2)
			<< linear_layer<relu>(128, 128)
			<< linear_layer<tan_h>(128, 65);

		assert(model.in_data_size() == 128);
		assert(model.out_data_size() == 128);

		optimizer.alpha = lr;
	}

	int compute_action(__Board bd) {
		vector<int> legalMoves = GenerateLegalMoveList(bd);
		/*if (rand_real(0.0, 1.0) < exploration_proba) {
			return legalMoves[rand_int(0, legalMoves.size() - 1)];
		}*/
		vector<vec_t> q_values = model.predict(ConvertState(bd));
		
		/*int ind = 0
		float_t max = 0;
		for (int i = 0; i < q_values.size(); i++) {
			if (max < q_values[i])
		}*/

		return 1;
	}
};