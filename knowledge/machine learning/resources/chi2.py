from sklearn.feature_selection import chi2

def chi2_select(X, y, X2, percentile=70):
    _, ncols = X.shape
    chi2_stat, _ = chi2(X, y)
    indexed_chi2 = list(zip(range(ncols), chi2_stat))
    indexed_chi2.sort(key=lambda x: x[1], reverse=True)
    col_index = [x[0] for x in indexed_chi2]
    length = int(percentile * ncols / 100)
    X = X[:, col_index[:length]]
    X2 = X2[:, col_index[:length]]
    return X, X2

def plot_chi2_curve(model, X, y, X2, y2, pc_range):
	train_scores = []
	dev_scores = []
	_, ncols = X.shape
    chi2_stat, _ = chi2(X, y)
    indexed_chi2 = list(zip(range(ncols), chi2_stat))
    indexed_chi2.sort(key=lambda x: x[1], reverse=True)
    col_index = [x[0] for x in indexed_chi2]
    for percentile in pc_range:
    	length = int(percentile * ncols / 100)
    	X_ = X[:, col_index[:length]]
    	X2_ = X2[:, col_index[:length]]
    	model.fit(X_, y)
    	train_scores.append(model.score(X_, y))
    	dev_scores.append(model.score(X2_, y2))
	plt.plot(pc_range, train_scores, lw=2, color='darkorange', label='train')
	plt.plot(pc_range, dev_scores, lw=2, color='darkorange', label='test')
	plt.ylim(0.0, 1.1)
    plt.xlabel('percentile')
	plt.ylabel('score')
	plt.legend(loc='best')
	# plt.show()