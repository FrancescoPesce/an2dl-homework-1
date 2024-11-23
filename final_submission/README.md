Submission for the first homework of AN2DL by group LosPollosHermanos (Mohammadhossein Allahakbari, Michele Miotti, Francesco Pesce).

Content index:
- `report.pdf`: The report for the homework.
- `notebook.ipynb`: The notebook used to define and train the final model. It also contains some visualizations and some techniques used in older models no longer employed in the latest one.
- `averager.py`: A python script to average the weights of some models. No longer used but included since in improved the accuracy of our models for a couple of models, as explained in the report.
- `overlay.png`, `model.png`, `model.py`, `__pychache__`: Resources and outputs of `notebook.ipynb`.
- `submission_241123_000705.zip`: Final submission on Codabench, it is also an output of `notebook.ipynb`.

Note: this folder does not include `training_set.npz` to reduce its memory size. If you wish to execute `notebook.ipynb`, please add said file to this directory. The folder does not include the model's weights at each epoch, generated and saved by `notebook.ipynb`, for the same reason. Clearly, it does include the weights at the final epoch.