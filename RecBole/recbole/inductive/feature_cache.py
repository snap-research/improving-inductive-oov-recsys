class InductiveFeatureCache:
    """A variable that can be passed to inductive embedders to cache the features of users and items.
    This prevents the need to store the features in each embedder.
    """

    def __init__(self, mode='transductive'):
        self._user_feats = None
        self._item_feats = None
        self.mode = mode

    def get_mode(self):
        return self.mode

    def has_cached(self):
        return self._user_feats is not None and self._item_feats is not None

    def get_cached(self):
        return self._user_feats, self._item_feats

    def add_to_cache(self, user_feats, item_feats):
        self._user_feats = user_feats
        self._item_feats = item_feats
