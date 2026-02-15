# metrics_calculator.py
import numpy as np
import os

def calculate_and_print_metrics(results: list,output_dir: str = "results", filename: str = None):
	"""
	根据测试结果计算并打印所有指定的评估指标。

	Args:
		results (list): 包含每个谈判回合结果的字典列表。
	"""
	n_total = len(results)
	if n_total == 0:
		print("没有测试结果可供分析。")
		return

	successful_negotiations = [r for r in results if r['success'] and r['final_price'] >= r['seller_bottom_price'] and r['final_price'] <= r['buyer_max_price']]
	n_successful = len(successful_negotiations)

	# 1. Negotiation Success Rate (NSR)
	nsr = n_successful / n_total if n_total > 0 else 0.0
	os.makedirs(output_dir, exist_ok=True)
	output_path = os.path.join(output_dir, filename)
	if n_successful == 0:
		output_lines = [
			f"总谈判次数: {n_total}",
			f"谈判成功率 (NSR): {nsr:.2%}",
			"\n由于没有成功的谈判，无法计算其他指标。"
		]

		# 保存到文件
		with open(output_path, 'w', encoding='utf-8') as f:
			f.write("\n".join(output_lines))
		return

	gain_rates_b = []
	gain_rates_s = []
	bias_values = []
	turn_value = []
	for res in successful_negotiations:
		p = res['final_price']
		p_bar_b = res['buyer_max_price']
		#p_b = res['initial_buyer_price']
		#p_bar_s = res['initial_seller_price']
		p_s = res['seller_bottom_price']
		# --- 新增：获取数量，如果不存在则默认为1 ---
		#q = res.get('quantity', 1)
		# 2. Gain Rate (GR)
		gr_b = (p_bar_b - p) / (p_bar_b - p_s) if (p_bar_b - p_s) != 0 else 0
		gr_s = (p - p_s) / (p_bar_b - p_s) if (p_bar_b - p_s) != 0 else 0

		gain_rates_b.append(np.clip(gr_b, 0, 1))
		gain_rates_s.append(np.clip(gr_s, 0, 1))

		# 4. Bias Value (BV)
		bv = gr_s - gr_b
		bias_values.append(bv)
		turn_value.append(res['turns'])

	# Average Gain Rates
	gr_tilde_b = np.sum(gain_rates_b) / n_total
	gr_tilde_s = np.sum(gain_rates_s) / n_total

	# 3. Combined and Difference Gain Rates (CGR, DGR)
	cgr = np.mean(np.array(gain_rates_b) * np.array(gain_rates_s))
	dgr = np.mean(np.abs(np.array(gain_rates_b) - np.array(gain_rates_s)))

	# Average Bias Value and Standard Deviation
	bv_tilde = np.mean(bias_values)
	sigma_bv = np.std(bias_values)

	turn_rate = np.mean(turn_value)
	# --- 构建输出内容 ---
	output_lines = [
		"\n" + "=" * 50,
		"谈判测试结果分析",
		"=" * 50,
		f"总谈判次数: {n_total}",
		f"成功谈判次数: {n_successful}",
		"-" * 50,
		f"1. 谈判成功率 (NSR): {nsr:.2%}",
		"-" * 50,
		"2. 平均收益率 (GR):",
		f"   - 买家平均收益率 (GR_b): {gr_tilde_b:.4f}",
		f"   - 卖家平均收益率 (GR_s): {gr_tilde_s:.4f}",
		"-" * 50,
		"3. 平衡性指标:",
		f"   - 综合收益率 (CGR): {cgr:.4f} (越接近 0.25 越平衡)",
		f"   - 差异收益率 (DGR): {dgr:.4f} (越接近 0 越平衡)",
		"-" * 50,
		"4. 偏向性指标 (BV):",
		f"   - 平均偏向值 (BV_tilde): {bv_tilde:.4f} (>0 偏向卖家, <0 偏向买家)",
		f"   - 偏向值标准差 (σ(BV)): {sigma_bv:.4f} (越小越公平/稳定)",
		"=" * 50 + "\n"
		"5. 平均轮次：",
		f"	-平均谈判轮次(turn_rate): {turn_rate:.4f}",
		"=" * 50 + "\n"
	]
	with open(output_path, 'w', encoding='utf-8') as f:
		f.write("\n".join(output_lines))

	print(f"\n结果已保存至: {output_path}")