import attrs

from qoc import struct

from ._abc import Feature


@attrs.define
class FeatureStore:
    """FeatureStore is a container for multiple features that can be computed together.

    Examples:
        >>> import qoc.feature
        >>> features = qoc.feature.FeatureStore([qoc.feature.FeaturePrice()])
        >>> state = features.compute(timestamp, state)
    """

    features: list[Feature] = attrs.field(factory=list)

    def compute(self, timestamp: struct.Timestamp, state: struct.State) -> struct.State:
        for feature in self.features:
            state = feature.compute(timestamp, state)
        return state
