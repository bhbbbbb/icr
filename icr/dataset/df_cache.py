from typing import Dict
import pandas as pd

class DfCache:

    def __init__(self):

        self._cache: Dict[str, pd.DataFrame] = {}
        return
    
    def get_copy(self, path: str):
        df = self._cache.get(path, None)
        if df is None:
            df = pd.read_csv(path)
            self._cache[path] = df
        return df.copy()
