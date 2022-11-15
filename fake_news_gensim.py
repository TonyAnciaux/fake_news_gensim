import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import spacy
import pickle


wv = api.load("word2vec-google-news-300")
nlp = spacy.load("en_core_web_lg")


def prepare_data(fakenews_path, realnews_path):
	fake = pd.read_csv(fakenews_path)
	true = pd.read_csv(realnews_path)

	fake["label"] = "fake"
	fake = fake[["text", "label"]]
	true["label"] = "real"
	true = true[["text", "label"]]

	df = pd.concat([true, fake])
	df["label_num"] = df["label"].map({"fake": 0, "real": 1})
	return df


def preprocess_and_vectorize(text):
	doc = nlp(text)
	filtered_token = list()
	for token in doc:
		if token.is_punct or token.is_stop:
			continue
		filtered_token.append(token.lemma_)
	return wv.get_mean_vector(filtered_token)


def fit_model_to_train_data(model, X_train, y_train, save=True):
	classifier = model()
	classifier.fit(X_train, y_train)
	if save:
		pickle.dump(model, open('model.pkl', 'wb'))
	return classifier


def predict(model, X_test):
	return model.predict(X_test)


def report_results(model, y_test, y_pred, visuals=True, title_name="Confusion Matrix", save=True, xlims=(-180, 180), ylims=(-180, 180)):
	if visuals:
		cm.confusion_matrix(y_test, y_pred)
		ax = sns.heatmap(cm, annot=True, fmt="d")
		ax.set_title(title_name)
		ax.set(xlim=xlims)
		ax.set(ylim=ylims)
		ax.set_xlabel('Predictions')
		ax.set_ylabel("True Labels")
		fig = ax.get_figure()
		if save:
			fig.savefig(title_name + "cfn_mtx.png")
		return fig

	else:
		return classification_report(y_test, y_pred)
	

if __name__ == "__main__":

	df = prepare_data("./Fake.csv", "./True.csv")
	df["vector"] = df["text"].apply(preprocess_and_vectorize)

	X_train, X_test, y_train, y_test = train_test_split(
		df["vector"].values,
		df["label_num"],
		test_size=0.2,
		random_state=42,
		stratify=df["label_num"]
		)

	# reshaping 1D vectors in 2D
	X_train_2d = np.stack(X_train)
	X_test_2d = np.stack(X_test)

	clf = fit_model_to_train_data(GradientBoostingClassifier, X_train, y_train)

	y_pred = clf.predict(clf, X_test_2d)
	graph = report_results(clf, y_test, y_pred)
	graph.show()
