#pragma once

#ifndef MEDIAN_BLUR
#define MEDIAN_BLUR

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <math.h>
using std::vector;
using std::string;
using std::accumulate;
using std::pow;

class Median_Blur {
public:
	vector<vector<unsigned> > get_median_image(vector<vector<unsigned>> image, unsigned radius, string padding_way) {
		unsigned H = image.size();  // rows
		unsigned W = image[0].size();  // cols
		vector<vector<unsigned>> img = get_padding_img(image, radius, padding_way);
		vector<vector<unsigned int> > H_hist;  //  = init_H_hist(radius); 
		vector<vector<unsigned> > h_hist = init_h_hist(img, W, radius);
		
		vector<vector<unsigned> > median_blur_image(H, vector<unsigned>(W));
		auto r = radius;
		// (i, j) is the position of current central pixel in image.
		for (int i = 0; i < H; ++i) {
			for (int j = 0; j < W; ++j) {
				int i_img = i + r;  // current i corresponding to i + r in padded image img
				int j_h = j + r;  // current j corresponding to j + r in h_hist
				/* 1. Update h_hist and H_hist: */
				if (i == 0) {  // first line processed separately
					H_hist.clear();
					for (int _ = j_h - r; _ < j_h + r + 1; ++_)
						H_hist.push_back(h_hist[_]);
				}
				else {
					vector<int> j_move_down;
					if (j == 0) {  // first element of each line processed separately
						for (int _ = j_h - r; _ < j_h + r + 1; ++_)  // (2*r+1) columns in total
							j_move_down.push_back(_);
						move_down(img, h_hist, i_img, j_move_down, radius);  // first (2*r+1) vectors of h_hist updated
						H_hist.clear();
						for (int _ = j_h - r; _ < j_h + r + 1; ++_)
							H_hist.push_back(h_hist[_]);  // H_hist updated
					}
					else {
						j_move_down.push_back(j_h + r);
						move_down(img, h_hist, i_img, j_move_down, radius);  // next one column in h_hist updated
						move_right(H_hist, h_hist, j_h, radius);  // H_hist updated
					}
				}
				/* 2. Use updated H_hist to get median value to update median_image: */
				median_blur_image[i][j] = get_median(H_hist, radius);
			}
		}
		return median_blur_image;
	}

private:
	/* 0.1 定义padding函数 */
	vector<vector<unsigned> > get_padding_img(vector<vector<unsigned>> image, unsigned radius, string padding_way) {
		unsigned H = image.size();  // rows
		unsigned W = image[0].size();  // cols
		vector<vector<unsigned>> img(H + 2 * radius, vector<unsigned>(W + 2 * radius));
		 
		for (int row = 0; row < H + 2 * radius; ++row) {
			if (row < radius) {  // 上边边
				if (padding_way == "ZERO")
					img[row] = vector<unsigned>(W + 2 * radius, 0);
				else {  // padding_way == "REPLICATE"
					for (int col = 0; col < W + 2 * radius; ++col) {
						if (col < radius)  // 左边边
							img[row][col] = image[0][0];
						else if (col >= W + radius)  // 右边边
							img[row][col] = image[0][W - 1];
						else  // 中间
							img[row][col] = image[0][col - radius];
					}
				}
			}
			else if (row >= H + radius) {  // 下边边
				if (padding_way == "ZERO")
					img[row] = vector<unsigned>(W + 2 * radius, 0);
				else {  // padding_way == "REPLICATE"
					for (int col = 0; col < W + 2 * radius; ++col) {
						if (col < radius)  // 左边边
							img[row][col] = image[H - 1][0];
						else if (col >= W + radius)  // 右边边
							img[row][col] = image[H - 1][W - 1];
						else  // 中间
							img[row][col] = image[H - 1][col - radius];
					}
				}
			}
			else {  // 中间横域
				for (int col = 0; col < W + 2 * radius; ++col) {
					if (col < radius)  // 左边边
						img[row][col] = image[row - radius][0];
					else if (col >= W + radius)  // 右边边
						img[row][col] = image[row - radius][W - 1];
					else  // 中间
						img[row][col] = image[row - radius][col - radius];
				}
			}
		}

		return img;
	}
	/* 0.2 初始化H_hist矩阵 */
	/*vector<vector<unsigned> > init_H_hist(unsigned radius) {
		vector<vector<unsigned int> > H_hist;
		for (int i = 0; i < 2 * radius + 1; ++i)
			H_hist.push_back(vector<unsigned>(256, 0));
		return H_hist;
	}*/
	
	/* 1.1 定义get_hist函数 */
	vector<unsigned> get_hist(vector<vector<unsigned>> &img, int j, unsigned radius) {
		vector<unsigned> col;
		for (int i = 0; i < 2 * radius + 1; ++i)
			col.push_back(img[i][j]);
		vector<unsigned> hist(256, 0);
		for (auto val : col)
			++hist[val];
		return hist;
	}
	/* 1.2 初始化h_hist矩阵 */
	vector<vector<unsigned> > init_h_hist(vector<vector<unsigned>> &img, unsigned W, unsigned radius) {
		vector<vector<unsigned> > h_hist;
		for (int i = 0; i < W + 2 * radius; ++i)
			h_hist.push_back(get_hist(img, i, radius));
		return h_hist;
	}
	
	//------------------------------------------------------------
	// 到这里为止, 初始化的工作就结束了;下面定义的函数将在主循环中调用：
	/* 2. 定义位移函数: move_down(), move_right() */
	// First！
	// update h_hist:
	void move_down(vector<vector<unsigned>> &img, vector<vector<unsigned> > &h_hist, int i, vector<int> j_move, unsigned radius) {
		// i is the line of current central pixel in padded - image img;
		// j_move is the list of columns to be moved in h_hist.
		int i_remove = i - radius - 1;
		int i_add = i + radius;
		for (auto j : j_move) {
			auto remove_value = img[i_remove][j];
			auto add_value = img[i_add][j];
			--h_hist[j][remove_value];
			++h_hist[j][add_value];
		}
	}
	// Second!
	// update H_hist:
	void move_right(vector<vector<unsigned> > &H_hist, vector<vector<unsigned> > &h_hist, int j_h, unsigned radius) {
		// j_h is the column of current central pixel in h_hist.
		// 注意: 此时的H_hist尚未更新, 还处于上一个位置; 但j已经更新, 是当前位置的列.
		// j_remove = j_h - r - 1是相对于h_hist来说的, 事实上H_hist永远只需要del第0列.
		int j_add = j_h + radius;
		for (int i = 0; i < 2 * radius; ++i)
			H_hist[i] = H_hist[i + 1];  // 顺序往后一个:原首项被remove
		H_hist[2 * radius] = h_hist[j_add];
	}

	/* 3. 定义求中值函数: get_median() */
	/* 这里！！！！*/
	unsigned get_median(vector<vector<unsigned> > &H_hist, unsigned radius) {
		vector<unsigned> hist(256, 0);
		for (int val = 0; val < 256; ++val) {
			for(auto vec: H_hist)
				hist[val] += vec[val];  // modified
		}
				
		int thres = pow(2* radius +1, 2) / 2 + 1;
		int sum_cnt = 0;
		int median = 0;
		for (unsigned val = 0; val < 256; ++val) {
			auto cnt = hist[val];
			sum_cnt += cnt;
			if (sum_cnt >= thres) {
				median = val;
				break;
			}
		}
		return median;
	}
};

#endif