#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <forward_list>
using std::forward_list;
using std::min;
using std::max;
using std::sort;
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

	return s1 / s2;
}

vector<vector<float>> NMS(vector<vector<float>> lists, float thre) {
	// lists[0:4]: x1, x2, y1, y2; 
	// lists[4]: score
	sort(lists.begin(), lists.end(), [](const vector<float> &a, const vector<float> &b) { return a[4] > b[4]; });
	forward_list<vector<float>> B(lists.begin(), lists.end());
	vector<vector<float>> D;
	while(!B.empty()) {
		vector<float> MS = B.front();
		D.push_back(MS);
		B.pop_front();		
		
		auto prev = B.before_begin();
		auto curr = B.begin();
		while (curr != B.end()) {
			auto iou = IoU(MS, *curr);
			if (iou >= thre) {
				curr = B.erase_after(prev);
			}
			else {
				prev = curr;
			}
			++curr;
		}
	}
	return D;
}


