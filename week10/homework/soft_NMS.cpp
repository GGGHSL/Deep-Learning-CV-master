#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <forward_list>
using std::forward_list;
using std::min;
using std::max;
using std::cin;
using std::cout;
using std::endl;
using std::string;
using std::vector;

float IoU(vector<float>bi, vector<float>bj) {
	float x1 = bi[0], y1 = bi[1], x2 = bi[2], y2 = bi[3];
	float u1 = bj[0], v1 = bj[1], u2 = bj[2], v2 = bj[3];
	if (u1 >= x2 || u2 <= x1 || v1 >= y2 || v2 <= y1) {
		return 0;
	}
	// ¡É
	float w1 = min(x2, u2) - max(x1, u1);
	float h1 = min(y2, v2) - max(y1, v1);
	float s1 = h1 * w1;
	// ¡È
	float sbi = (x2 - x1)*(y2 - y1);
	float sbj = (u2 - u1)*(v2 - v1);
	float s2 = sbi + sbj - s1;
	// ¡É/¡È
	return s1 / s2;
}

vector<vector<float>> softNMS(vector<vector<float>> lists, float thre, float sigma) {
	// lists[0:4]: x1, x2, y1, y2; 
	// lists[4]: score
	vector<vector<float>> B(lists.begin(), lists.end());
	vector<vector<float>> D;
	while (!B.empty()) {
		sort(B.begin(), B.end(), [](const vector<float> &a, const vector<float> &b) { return a[4] < b[4]; });
		vector<float> M = B.back();
		D.push_back(M);
		B.pop_back();

		for (auto it = B.begin(); it != B.end();++it) {
			float iou = IoU(M, *it);
			if (iou >= thre) {
				(*it)[4] *= exp(-iou * iou / sigma);
			}
		}
	}
	return D;
}

int main() {
	vector<vector <float>> m;
	m.push_back(vector<float>{1, 1, 3, 4, 0.8});
	m.push_back(vector<float>{1.1, 1.1, 3.1, 4.1, 0.9});
	m.push_back(vector<float>{4, 5, 8, 2, 0.5});
	auto D = softNMS(m, 0.7, 1);
	for (auto vec : D) {
		for (auto i : vec) {
			cout << i << " ";
		}
		cout << endl;
	}

	system("pause");
	return 0;
}