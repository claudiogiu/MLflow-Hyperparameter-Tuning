preprocess:
	python src/preprocess.py
	@echo "Preprocessing completed."


tune_model: preprocess
	python src/tune_model.py
	@echo "Model tuning completed."


train_model: tune_model
	python src/train_model.py
	@echo "Model training completed."


evaluate_model: train_model
	python src/evaluate_model.py
	@echo "Model evaluation completed."


run_all: preprocess tune_model train_model evaluate_model
	@echo "All steps completed."
