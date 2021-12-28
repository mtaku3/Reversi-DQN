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

vec_t ConvertState(__Board board) {
	float data[2][64];
	board.piece_planes((char*)&data);

	vec_t res(128);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 64; j++) {
			res[i * 64 + j] = data[i][j];
		}
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

	network<sequential> nn1;
	network<sequential> nn2;
	vector<network<sequential>> nets;
	adam optimizer;

	struct ExperienceData {
		__Board current_state;
		int action;
		double reward;
		__Board next_state;
		bool done;
	};

	vector<ExperienceData> memory_buffer;
	int max_memory_buffer;
	int memory_buffer_ind;

	DQNAgent(bool train = true) {
		lr = 0.001;
		gamma = 0.99;
		exploration_proba = 1.0;
		exploration_proba_decay = 0.005;
		batch_size = 256;

		max_memory_buffer = 131072;
		memory_buffer_ind = 0;
		memory_buffer = vector<ExperienceData>(max_memory_buffer);

		net_phase netphase = train ? net_phase::train : net_phase::test;
		nn1 << convolutional_layer<relu>(8, 8, 3, 2, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase)
			<< convolutional_layer<relu>(8, 8, 3, 128, 128, tiny_dnn::padding::same)
			<< batch_normalization_layer(64, 128, 1e-5, 0.1, netphase);
		
		nn2 << fully_connected_layer<relu>(128 * 64, 128)
			<< fully_connected_layer<tan_h>(128, 65);

		nets.push_back(nn1);
		nets.push_back(nn2);

		optimizer.alpha = lr;
	}

	void loadParams() {
		for (size_t i = 0; i < 2; i++) {
			std::ostringstream modelPath;
			modelPath << "./" << std::setfill('0') << std::setw(2) << i + 1
				<< ".weights";
			std::ifstream ifs(modelPath.str());
			if (ifs.fail()) {
				std::cout << "Failed to load weights from " << modelPath.str()
					<< std::endl;
			}
			else {
				std::cout << "Loading weights from " << modelPath.str() << std::endl;
			}
			ifs >> nets[i];
		}
	}

	auto predict(vec_t data) {
		auto output1 = nets[0].predict(data);
		auto output2 = nets[1].predict(output1);
		return output2;
	}

	int compute_action(__Board bd, bool train = true) {
		vector<int> legalMoves = GenerateLegalMoveList(bd);
		/*if (train && rand_real(0.0, 1.0) < exploration_proba) {
			return legalMoves[rand_int(0, legalMoves.size() - 1)];
		}*/
		vec_t q_values = predict(ConvertState(bd));
		
		int action = legalMoves[0];
		float max = -1.0e6;
		for (auto i : legalMoves) {
			if (max < q_values[i]) {
				max = q_values[i];
				action = i;
			}
		}

		return action;
	}
	/*
	void update_exploration_probability() {
		exploration_proba = exploration_proba * exp(-exploration_proba_decay);
		cout << "exploration_proba = " << exploration_proba << endl;
	}

	void store_episode(__Board current_state, int action, double reward, __Board next_state, bool done) {
		memory_buffer[memory_buffer_ind++] = ExperienceData{ current_state, action, reward, next_state, done };

		if (max_memory_buffer - 1 <= memory_buffer_ind) {
			memory_buffer_ind = 0;
		}
	}

	void train() {
		shuffle(memory_buffer.begin(), memory_buffer.end(), engine);
		vector<ExperienceData> batch_sample;
		copy(memory_buffer.begin(), memory_buffer.begin() + batch_size, back_inserter(batch_sample));

		for (auto experience : batch_sample) {
			vec_t q_current_state = model.predict(ConvertState(experience.current_state));

			double q_target = experience.reward;
			if (!experience.done) {
				vec_t _q_next_state = model.predict(ConvertState(experience.next_state));
				q_target = q_target + gamma * (double)(*max_element(_q_next_state.begin(), _q_next_state.end()));
			}
			q_current_state[experience.action] = q_target;

			model.train<mse>(optimizer, vector<vec_t>{ ConvertState(experience.current_state) }, vector<vec_t>{ q_current_state });
		}
	}*/
};