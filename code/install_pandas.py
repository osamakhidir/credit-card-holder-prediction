"""Small validator script: checks for pandas, prints version and runs a tiny sample.

Run after activating your virtual environment:
	python install_pandas.py
"""

import importlib
import sys


def main():
	pkg = 'pandas'
	spec = importlib.util.find_spec(pkg)
	if spec is None:
		print('pandas is not installed. Install with: pip install pandas')
		sys.exit(2)

	import pandas as pd
	print('pandas version:', pd.__version__)

	# small smoke test: create a DataFrame and print
	df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
	print('Sample DataFrame:')
	print(df)


if __name__ == '__main__':
	main()
pip install pandas